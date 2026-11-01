"""Calibrate VO/VI queueing onset using explicit per-STA offered rates."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, backend_source_hash, load_config
from rl.artifacts import atomic_output, save_json
from rl.gates import require_edca_validation
from rl.trainer import save_csv
from sim.backend.base_backend import ControlAction
from sim.backend.ns3_backend import Ns3Backend
from sim.statistics.metrics import aggregate_accounting, summarize


AC_INDEX = {"vo": 0, "vi": 1}
RATES = {"vo": (0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00),
         "vi": (0.25, 0.50, 1.00, 1.50, 2.00, 2.50, 3.00)}
SEEDS = (501, 502, 503)
WARMUP_S = 10
MEASURE_S = 20


def case_name(ac: str, per_sta_mbps: float, seed: int) -> str:
    rate = str(per_sta_mbps).replace(".", "p")
    return f"{ac}_per_sta_{rate}mbps_seed{seed}"


def run_case(job: tuple[dict, str, float, int, str]) -> dict:
    config, ac, per_sta_mbps, seed, output = job
    directory = Path(output) / case_name(ac, per_sta_mbps, seed)
    directory.mkdir(parents=True, exist_ok=False)
    ac_index = AC_INDEX[ac]
    mix = [0.0] * 4; mix[ac_index] = 1.0
    scenario = dict(offered_mbps=per_sta_mbps * config["network"]["num_sta"],
                    ac_mix=mix, ul_fraction=0.5, arrival="poisson")
    started = time.monotonic(); action = ControlAction()
    measure_rows = []; delays = []; pending_series = []
    queue_limit = config["queues"]["limits_packets"][ac_index]
    near_full = 0; queue_snapshots = 0
    with Ns3Backend(config, scenario, seed, directory, trace=False) as backend:
        for _ in range(WARMUP_S):
            events = backend.step(action, 1.0)
            summarize(events, config["network"]["num_sta"], 1.0)
        for second in range(MEASURE_S):
            events = backend.step(action, 1.0)
            metrics = summarize(events, config["network"]["num_sta"], 1.0)
            measure_rows.append(dict(second=second + 1, **metrics))
            delays.extend(event.values[4] for event in events
                          if event.kind == "D" and int(event.values[2]) == ac_index)
            records = [event.values for event in events
                       if event.kind == "P" and int(event.values[2]) == ac_index]
            pending = int(sum(record[10] for record in records))
            pending_series.append(pending)
            near_full += sum(record[10] >= 0.9 * queue_limit for record in records)
            queue_snapshots += len(records)
        if abs(backend.time - (WARMUP_S + MEASURE_S)) > 1e-9:
            raise RuntimeError(f"Unexpected stop time {backend.time}")
    accounting = aggregate_accounting(measure_rows)
    offered_mbps = sum(row[f"{ac}_offered_mbps"] for row in measure_rows) / MEASURE_S
    goodput_mbps = sum(row[f"{ac}_goodput_mbps"] for row in measure_rows) / MEASURE_S
    result = dict(ac=ac, per_sta_target_mbps=per_sta_mbps,
                  network_target_mbps=scenario["offered_mbps"], seed=seed,
                  queue_limit_packets=queue_limit,
                  warmup_s=WARMUP_S, measure_s=MEASURE_S,
                  offered_mbps=offered_mbps, goodput_mbps=goodput_mbps,
                  throughput_ratio=goodput_mbps / max(offered_mbps, 1e-12),
                  drop_rate=accounting[f"{ac}_drop_rate"],
                  overflow_count=accounting[f"{ac}_overflow_count"],
                  expired_count=accounting[f"{ac}_expired_count"],
                  failed_count=accounting[f"{ac}_failed_count"],
                  other_drop_count=accounting[f"{ac}_other_drop_count"],
                  pending_start=accounting[f"{ac}_pending_packets_start"],
                  pending_end=accounting[f"{ac}_pending_packets_end"],
                  pending_mean=float(np.mean(pending_series)),
                  pending_max=max(pending_series),
                  near_full_snapshot_fraction=near_full / max(queue_snapshots, 1),
                  delay_samples=len(delays),
                  delay_mean_ms=float(np.mean(delays)) if delays else 0.0,
                  delay_p95_ms=float(np.percentile(delays, 95)) if delays else 0.0,
                  delay_p99_ms=float(np.percentile(delays, 99)) if delays else 0.0,
                  delay_p99_9_ms=float(np.percentile(delays, 99.9)) if delays else 0.0,
                  delay_max_ms=float(np.max(delays)) if delays else 0.0,
                  runtime_s=time.monotonic() - started)
    with atomic_output(directory / "delay_samples.npz", binary=True) as stream:
        np.savez_compressed(stream, **{ac: np.asarray(delays, dtype=np.float64)})
    save_csv(directory / "periods.csv", measure_rows)
    save_json(directory / "summary.json", result)
    save_json(directory / "scenario.json", scenario)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    gate = require_edca_validation(); config = load_config(ROOT / "configs/full.yaml")
    capacity = json.loads((ROOT / config["traffic"]["capacity_file"]).read_text(encoding="utf-8"))
    if gate["source_hash"] != capacity["source_hash"] or gate["source_hash"] != backend_source_hash():
        raise RuntimeError("Backend validation/calibration fingerprint mismatch")
    jobs = [(config, ac, rate, seed, str(args.output))
            for ac, rates in RATES.items() for rate in rates for seed in SEEDS]
    metadata = dict(state="running", source_hash=gate["source_hash"], seeds=SEEDS,
                    rates=RATES, warmup_s=WARMUP_S, measure_s=MEASURE_S,
                    queue_limits=config["queues"]["limits_packets"],
                    queue_lifetime_s=config["queues"]["max_delay_s"],
                    retry_limit=config["network"]["retry_limit"], jobs=len(jobs),
                    started_at=datetime.now().astimezone().isoformat())
    save_json(args.output / "metadata.json", metadata)
    rows = []; started = time.monotonic()
    with ProcessPoolExecutor(max_workers=config["run"]["workers"]) as pool:
        futures = {pool.submit(run_case, job): (job[1], job[2], job[3]) for job in jobs}
        for future in as_completed(futures):
            row = future.result(); rows.append(row)
            save_csv(args.output / "runs.csv", sorted(rows, key=lambda r: (r["ac"], r["per_sta_target_mbps"], r["seed"])))
            print(json.dumps({"complete": futures[future], "drop_rate": row["drop_rate"],
                              "p99_9_ms": row["delay_p99_9_ms"],
                              "pending_end": row["pending_end"]}), flush=True)
    metadata.update(state="complete", runtime_s=time.monotonic() - started,
                    completed_at=datetime.now().astimezone().isoformat())
    save_json(args.output / "metadata.json", metadata)
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
