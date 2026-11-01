"""Paired Default STEP calibration for direct-MAC CMDP budgets."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config
from rl.tgax_environment import TgaxControlEnvironment


CONFIG = ROOT / "configs/mac_arrival40_gaming160_cmdp.yaml"
OUTPUT = ROOT / "results/raw/mac_arrival40_gaming160_cmdp_default_calibration_20261029"


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def run_seed(seed: int, duration: int) -> dict:
    config = load_config(CONFIG)
    directory = OUTPUT / f"seed_{seed}"
    if directory.exists() and (directory / "summary.json").is_file():
        return json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    started = time.monotonic()
    env = TgaxControlEnvironment(config, "compressed_async_age", seed, directory, float(duration))
    initial_debt = env.demand_debt.copy()
    initial_sta_debt = env.sta_demand_debt.copy()
    rows = []
    action = np.asarray(config["action"]["default_action"], dtype=np.int64)
    try:
        for step in range(duration):
            _, reward, metrics = env.step(action, 1.0, custom=True)
            rows.append({"step": step + 1, "reward": reward, **metrics})
    finally:
        env.close()
    generated = initial_debt + np.sum([row["application_generated_bytes"] for row in rows], axis=0)
    acked = np.sum([row["application_received_bytes"] for row in rows], axis=0)
    satisfaction = np.divide(acked, generated, out=np.zeros(4), where=generated > 0).clip(0, 1)
    sta_generated = initial_sta_debt + np.sum([row["sta_application_generated_bytes"] for row in rows], axis=0)
    sta_acked = np.sum([row["sta_application_received_bytes"] for row in rows], axis=0)
    sta_sat = np.divide(sta_acked, sta_generated, out=np.zeros(40), where=sta_generated > 0).clip(0, 1)
    active = sta_generated > 0; values = sta_sat[active]
    jain = float(values.sum() ** 2 / (len(values) * np.square(values).sum())) if np.square(values).sum() else 0.0
    eligible = {ac: sum(row[f"{ac}_deadline_eligible"] for row in rows) for ac in ("vo", "vi")}
    ontime = {ac: sum(row[f"{ac}_on_time_delivered"] for row in rows) for ac in ("vo", "vi")}
    summary = {"seed": seed, "duration_s": duration, "wall_clock_s": time.monotonic() - started,
               "generated_bytes": generated.tolist(), "acked_bytes": acked.tolist(),
               "satisfaction": satisfaction.tolist(), "jain": jain,
               "deadline_eligible": eligible, "on_time": ontime,
               "timely": {ac: ontime[ac] / eligible[ac] if eligible[ac] else None for ac in eligible},
               "mean_step_costs": np.mean([row["cmdp_costs"] for row in rows], axis=0).tolist(),
               "mean_step_satisfaction": np.mean(
                   [row["application_satisfaction"] for row in rows], axis=0).tolist(),
               "mean_step_jain": float(np.mean([row["jain_demand_satisfaction"] for row in rows])),
               "mean_valid_step_timely": {
                   ac: float(np.mean([row[f"{ac}_on_time_delivery_rate"] for row in rows
                                      if row[f"{ac}_on_time_delivery_valid"]]))
                       if any(row[f"{ac}_on_time_delivery_valid"] for row in rows) else None
                   for ac in ("vo", "vi")},
               "all_finite": bool(np.isfinite(np.asarray([row["reward"] for row in rows])).all())}
    with (directory / "steps.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("step", "reward", "cmdp_utility", "cmdp_costs", "application_generated_bytes",
                  "application_received_bytes", "application_demand_debt_bytes",
                  "jain_demand_satisfaction", "vo_deadline_eligible", "vo_on_time_delivered",
                  "vi_deadline_eligible", "vi_on_time_delivered")
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key]) if isinstance(row[key], list) else row[key] for key in fields})
    atomic_json(directory / "summary.json", summary)
    return summary


def main() -> None:
    global CONFIG, OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    CONFIG = args.config if args.config.is_absolute() else ROOT / args.config
    OUTPUT = args.output if args.output.is_absolute() else ROOT / args.output
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUTPUT / "status.json", {"state": "running", "seeds": args.seeds,
                "duration_s": args.duration, "completed": []})
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_seed, seed, args.duration): seed for seed in args.seeds}
        for future in as_completed(futures):
            results.append(future.result())
            atomic_json(OUTPUT / "status.json", {"state": "running", "seeds": args.seeds,
                        "duration_s": args.duration, "completed": sorted(row["seed"] for row in results)})
    results.sort(key=lambda row: row["seed"])
    eligible = {ac: sum(row["deadline_eligible"][ac] for row in results) for ac in ("vo", "vi")}
    ontime = {ac: sum(row["on_time"][ac] for row in results) for ac in ("vo", "vi")}
    timely = {ac: ontime[ac] / eligible[ac] for ac in eligible}
    mean_sat = np.mean([row["satisfaction"] for row in results], axis=0)
    mean_jain = float(np.mean([row["jain"] for row in results]))
    # Budgets are frozen on the same one-second scale used by multiplier
    # updates.  Cumulative end-of-run values are retained above for audit only.
    mean_step_sat = np.mean([row["mean_step_satisfaction"] for row in results], axis=0)
    mean_step_jain = float(np.mean([row["mean_step_jain"] for row in results]))
    mean_valid_timely = {
        ac: float(np.mean([row["mean_valid_step_timely"][ac] for row in results
                           if row["mean_valid_step_timely"][ac] is not None]))
        for ac in ("vo", "vi")}
    low_targets = [0.8 * float(mean_step_sat[2]), 0.8 * float(mean_step_sat[3])]
    budgets = [1 - 0.95 * mean_valid_timely["vo"],
               1 - 0.95 * mean_valid_timely["vi"],
               1 - 0.95 * mean_step_jain, 1.0]
    aggregate = {"status": "PASS" if all(row["all_finite"] for row in results) else "FAIL",
                 "seeds": args.seeds, "duration_s": args.duration, "per_seed": results,
                 "aggregate_timely": timely, "mean_satisfaction": mean_sat.tolist(),
                 "mean_jain": mean_jain,
                 "mean_valid_step_timely": mean_valid_timely,
                 "mean_step_satisfaction": mean_step_sat.tolist(),
                 "mean_step_jain": mean_step_jain,
                 "recommended_low_priority_min_satisfaction": low_targets,
                 "recommended_budgets": budgets,
                 "rules": {"deadline_and_jain": "95% of Default on the one-second multiplier-update scale",
                           "be_bk": "80% of Default one-second demand satisfaction",
                           "cumulative_metrics": "audit only; not mixed into step-scale budgets"},
                 "config": str(CONFIG.relative_to(ROOT)).replace("\\", "/")}
    atomic_json(OUTPUT / "calibration.json", aggregate)
    atomic_json(OUTPUT / "status.json", {"state": "complete", **aggregate})
    print(json.dumps(aggregate, indent=2))
    if aggregate["status"] != "PASS": raise RuntimeError("CMDP calibration failed")


if __name__ == "__main__":
    main()
