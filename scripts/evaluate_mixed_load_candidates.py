"""Evaluate absolute per-STA mixed-AC candidates before changing official scenarios."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config
from rl.artifacts import atomic_output, save_json
from rl.gates import require_edca_validation
from rl.trainer import save_csv
from sim.backend.base_backend import ControlAction
from sim.backend.ns3_backend import Ns3Backend
from sim.statistics.metrics import aggregate_accounting, summarize


OUTPUT = ROOT / "results/raw/mixed_load_candidates_20261019"
SEEDS = (501, 502, 503)
WARMUP_S = 10
MEASURE_S = 20
ACS = ("vo", "vi", "be", "bk")
# Aggregate Mbps per STA across UL and DL. Each direction receives half.
CANDIDATES = {
    "current_S2": (0.3481036875, 0.8122419375, 1.0443110625, 0.1160345625),
    "current_S4": (0.2621016, 0.7207794, 1.965762, 0.327627),
    "realtime_edge": (0.75, 2.00, 0.25, 0.05),
    "realtime_tail": (1.00, 2.50, 0.25, 0.05),
    "mixed_overload": (1.00, 2.50, 1.00, 0.25),
}


def run_case(job: tuple[dict, str, tuple[float, ...], int, str]) -> dict:
    config, name, rates, seed, output = job
    directory = Path(output) / f"{name}_seed{seed}"; directory.mkdir(parents=True, exist_ok=False)
    per_sta_total = sum(rates)
    scenario = dict(offered_mbps=per_sta_total * config["network"]["num_sta"],
                    ac_mix=[value / per_sta_total for value in rates],
                    ul_fraction=0.5, arrival="poisson")
    action = ControlAction(); delays = [[] for _ in ACS]; rows = []
    pending = {ac: [] for ac in ACS}; started = time.monotonic()
    with Ns3Backend(config, scenario, seed, directory, trace=False) as backend:
        for _ in range(WARMUP_S):
            summarize(backend.step(action, 1.0), config["network"]["num_sta"], 1.0)
        for second in range(MEASURE_S):
            events = backend.step(action, 1.0)
            metrics = summarize(events, config["network"]["num_sta"], 1.0)
            rows.append(dict(second=second + 1, **metrics))
            for event in events:
                if event.kind == "D": delays[int(event.values[2])].append(event.values[4])
            for index, ac in enumerate(ACS):
                pending[ac].append(int(sum(event.values[10] for event in events
                                           if event.kind == "P" and int(event.values[2]) == index)))
    accounting = aggregate_accounting(rows)
    result = dict(candidate=name, seed=seed, per_sta_total_mbps=per_sta_total,
                  network_target_mbps=scenario["offered_mbps"], warmup_s=WARMUP_S,
                  measure_s=MEASURE_S, runtime_s=time.monotonic() - started)
    for ac, rate, samples in zip(ACS, rates, delays):
        result[f"{ac}_per_sta_target_mbps"] = rate
        result[f"{ac}_offered_mbps"] = sum(row[f"{ac}_offered_mbps"] for row in rows) / MEASURE_S
        result[f"{ac}_goodput_mbps"] = sum(row[f"{ac}_goodput_mbps"] for row in rows) / MEASURE_S
        result[f"{ac}_drop_rate"] = accounting[f"{ac}_drop_rate"]
        for reason in ("overflow", "expired", "failed", "other_drop"):
            result[f"{ac}_{reason}_count"] = accounting[f"{ac}_{reason}_count"]
        result[f"{ac}_pending_start"] = accounting[f"{ac}_pending_packets_start"]
        result[f"{ac}_pending_end"] = accounting[f"{ac}_pending_packets_end"]
        result[f"{ac}_pending_mean"] = float(np.mean(pending[ac]))
        result[f"{ac}_pending_max"] = max(pending[ac])
        for label, quantile in (("p95", 95), ("p99", 99), ("p99_9", 99.9)):
            result[f"{ac}_delay_{label}_ms"] = float(np.percentile(samples, quantile)) if samples else 0.0
        result[f"{ac}_delay_max_ms"] = float(np.max(samples)) if samples else 0.0
    result.update(offered_packets=accounting["offered_packets"],
                  delivered_packets=accounting["delivered_packets"],
                  overflow_count=accounting["overflow_count"],
                  expired_count=accounting["expired_count"],
                  failed_count=accounting["failed_count"],
                  pending_start=accounting["pending_packets_start"],
                  pending_end=accounting["pending_packets_end"],
                  drop_rate=accounting["drop_rate"],
                  goodput_mbps=sum(row["goodput_mbps"] for row in rows) / MEASURE_S)
    with atomic_output(directory / "delay_samples.npz", binary=True) as stream:
        np.savez_compressed(stream, **{ac: np.asarray(samples, dtype=np.float64)
                                      for ac, samples in zip(ACS, delays)})
    save_csv(directory / "periods.csv", rows)
    save_json(directory / "summary.json", result)
    save_json(directory / "scenario.json", scenario)
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    gate = require_edca_validation(); config = load_config(ROOT / "configs/full.yaml")
    jobs = [(config, name, rates, seed, str(OUTPUT)) for name, rates in CANDIDATES.items() for seed in SEEDS]
    metadata = dict(state="running", source_hash=gate["source_hash"], candidates=CANDIDATES,
                    seeds=SEEDS, warmup_s=WARMUP_S, measure_s=MEASURE_S,
                    queue_limits=config["queues"]["limits_packets"],
                    queue_lifetime_s=config["queues"]["max_delay_s"],
                    retry_limit=config["network"]["retry_limit"],
                    started_at=datetime.now().astimezone().isoformat())
    save_json(OUTPUT / "metadata.json", metadata)
    rows = []; started = time.monotonic()
    with ProcessPoolExecutor(max_workers=config["run"]["workers"]) as pool:
        futures = {pool.submit(run_case, job): (job[1], job[3]) for job in jobs}
        for future in as_completed(futures):
            row = future.result(); rows.append(row)
            save_csv(OUTPUT / "runs.csv", sorted(rows, key=lambda r: (r["candidate"], r["seed"])))
            print(json.dumps({"complete": futures[future], "drop_rate": row["drop_rate"],
                              "vo_p99_9_ms": row["vo_delay_p99_9_ms"],
                              "vi_p99_9_ms": row["vi_delay_p99_9_ms"]}), flush=True)
    metadata.update(state="complete", runtime_s=time.monotonic() - started,
                    completed_at=datetime.now().astimezone().isoformat())
    save_json(OUTPUT / "metadata.json", metadata)


if __name__ == "__main__":
    main()
