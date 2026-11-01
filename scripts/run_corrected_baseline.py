"""Run the complete baseline matrix with the active exact-accounting schema."""
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
from sim.scenarios.resolver import resolve_scenario


ACS = ("vo", "vi", "be", "bk")


def run_case(job: tuple[dict, str, dict, int, float, float, str]) -> tuple[dict, list[dict]]:
    config, name, scenario, seed, warmup, duration, output = job
    directory = Path(output) / f"{name}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    period = config["timing"]["decision_s"]
    rows: list[dict] = []
    delays = [[] for _ in range(4)]
    station_bytes = np.zeros(config["network"]["num_sta"])
    with Ns3Backend(config, scenario, seed, directory, trace=False) as backend:
        warmup_end=backend.time+warmup
        while backend.time < warmup_end - 1e-9:
            backend.step(ControlAction(),min(period,warmup_end-backend.time))
        measurement_start=backend.time; measurement_end=measurement_start+duration
        while backend.time < measurement_end - 1e-9:
            step = min(period, measurement_end - backend.time)
            events = backend.step(ControlAction(), step)
            metrics = summarize(events, config["network"]["num_sta"], step)
            rows.append(dict(scenario=name, eval_seed=seed, time_s=backend.time-measurement_start, **metrics))
            for event in events:
                if event.kind == "D":
                    delays[int(event.values[2])].append(event.values[4])
                    station_bytes[int(event.values[1])] += event.values[3]
        if abs(backend.time - measurement_end) > 1e-9:
            raise RuntimeError(f"Unexpected stop time {backend.time}")
    with atomic_output(directory / "delay_samples.npz", binary=True) as stream:
        np.savez_compressed(stream, **{ac: np.asarray(values, dtype=np.float64)
                                      for ac, values in zip(ACS, delays)})
    save_csv(directory / "periods.csv", rows)
    summary = {key: float(np.mean([row[key] for row in rows]))
               for key, value in rows[0].items()
               if key not in ("scenario", "eval_seed", "time_s")
               and isinstance(value, (int, float, bool, np.number))}
    summary.update(method="default", scenario=name, eval_seed=seed,
                   traffic_warmup_s=warmup,duration_s=duration,target_offered_mbps=scenario["offered_mbps"])
    for ac, values in zip(ACS, delays):
        summary[f"{ac}_delay_mean_ms"] = float(np.mean(values)) if values else 0.0
        summary[f"{ac}_delay_p95_ms"] = float(np.percentile(values, 95)) if values else 0.0
        summary[f"{ac}_delay_p99_ms"] = float(np.percentile(values, 99)) if values else 0.0
        summary[f"{ac}_delay_p99_9_ms"] = float(np.percentile(values, 99.9)) if values else 0.0
        summary[f"{ac}_delay_max_ms"] = float(np.max(values)) if values else 0.0
        summary[f"{ac}_delay_samples"] = len(values)
    total = station_bytes.sum()
    summary["jain_fairness"] = float(total ** 2 / (len(station_bytes) * np.square(station_bytes).sum())) if total else 1.0
    summary.update(aggregate_accounting(rows))
    summary["goodput_mbps"] = sum(row["goodput_mbps"] for row in rows) / duration
    summary["runtime_s"] = time.monotonic() - started
    save_json(directory / "summary.json", summary)
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/full.yaml")
    parser.add_argument("--seeds",type=int,nargs="+")
    parser.add_argument("--duration",type=float)
    parser.add_argument("--warmup",type=float)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gate = require_edca_validation()
    config = load_config(args.config.resolve())
    if args.seeds: config["evaluation"]["seeds"]=args.seeds
    capacity = json.loads((ROOT / config["traffic"]["capacity_file"]).read_text(encoding="utf-8"))
    if capacity["source_hash"] != gate["source_hash"] or gate["source_hash"] != backend_source_hash():
        raise RuntimeError("Validation or capacity fingerprint does not match the backend")
    scenarios = load_config(ROOT / config["traffic"]["scenarios"])
    duration = args.duration if args.duration is not None else config["evaluation"]["duration_s"]
    warmup = args.warmup if args.warmup is not None else config["evaluation"].get("traffic_warmup_s",0.)
    jobs = []
    for name, definition in scenarios.items():
        scenario = resolve_scenario(definition,config["network"]["num_sta"],capacity["C_ref_mbps"])
        for seed in config["evaluation"]["seeds"]:
            jobs.append((config, name, scenario, seed, warmup, duration, str(args.output)))
    summaries: list[dict] = []
    periods: list[dict] = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=config["run"]["workers"]) as pool:
        futures = {pool.submit(run_case, job): (job[1], job[3]) for job in jobs}
        for future in as_completed(futures):
            summary, rows = future.result()
            summaries.append(summary); periods.extend(rows)
            save_csv(args.output / "evaluation.csv", sorted(summaries, key=lambda r: (r["scenario"], r["eval_seed"])))
            save_csv(args.output / "periods.csv", sorted(periods, key=lambda r: (r["scenario"], r["eval_seed"], r["time_s"])))
            print(json.dumps({"complete": futures[future], "drop_rate": summary["drop_rate"],
                              "vo_p99_9_ms": summary["vo_delay_p99_9_ms"]}), flush=True)
    versions={int(r["accounting_version"]) for r in summaries}
    if len(versions)!=1: raise RuntimeError(f"Mixed accounting versions: {versions}")
    metadata = dict(state="complete", method="default", accounting_version=versions.pop(),
                    source_hash=gate["source_hash"], capacity_mbps=capacity["C_ref_mbps"],
                    scenarios=list(scenarios), seeds=config["evaluation"]["seeds"],
                    traffic_warmup_s=warmup,duration_s=duration,jobs=len(jobs),runtime_s=time.monotonic()-started,
                    completed_at=datetime.now().astimezone().isoformat(),
                    delay_definition="generated before MAC Send to successful ACK; measurement-window departures only; successful packets only",
                    count_definition="exact packet outcomes plus start/end pending snapshots")
    save_json(args.output / "metadata.json", metadata)
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
