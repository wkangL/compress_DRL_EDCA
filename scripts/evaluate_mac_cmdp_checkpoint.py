#!/usr/bin/env python3
"""Resumable paired Default/final evaluation of an existing CMDP checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config
from scripts.run_tgax_ddqn_extended import evaluate_one, load_snapshot_model, save_csv


def existing_summary(directory: Path) -> dict | None:
    path = directory / "evaluation_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    snapshot = args.snapshot if args.snapshot.is_absolute() else ROOT / args.snapshot
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.duration <= 0 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("duration must be positive and seeds must be unique")
    if not snapshot.is_file():
        raise FileNotFoundError(snapshot)
    config = load_config(config_path)
    config["evaluation"]["duration_s"] = float(args.duration)
    try:
        checkpoint_step = int(snapshot.stem.rsplit("_", 1)[-1])
    except ValueError:
        checkpoint_step = None
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    status_path.write_text(json.dumps({"state": "running", "duration_s": args.duration,
                                       "seeds": args.seeds, "snapshot": str(snapshot)}, indent=2), encoding="utf-8")
    model, scaler = load_snapshot_model(snapshot, config)
    rows = []
    try:
        for seed in args.seeds:
            print(json.dumps({"phase": "evaluation_seed_start", "seed": seed,
                              "duration_s": args.duration}), flush=True)
            default_dir = output / f"seed_{seed}" / "default"
            final_dir = output / f"seed_{seed}" / "final"
            default = existing_summary(default_dir)
            if default is None:
                default, _ = evaluate_one(config, model, scaler, seed, default_dir, False)
            final = existing_summary(final_dir)
            if final is None:
                final, _ = evaluate_one(config, model, scaler, seed, final_dir, True)
            base_reward = float(default["reward_mean"])
            rows.append({"eval_seed": seed, "policy": "final", "checkpoint_step": checkpoint_step,
                         "duration_s": args.duration, "default_reward": base_reward,
                         "learned_reward": final["reward_mean"],
                         "reward_change_percent": ((float(final["reward_mean"])/base_reward-1)*100
                                                   if base_reward else None),
                         "paired_generated_demand_equal_each_step":
                             default["application_generated_by_step"] == final["application_generated_by_step"]})
            save_csv(output / "comparison.csv", rows)
            print(json.dumps({"phase": "evaluation_seed_complete", "seed": seed}), flush=True)
    except BaseException as error:
        status_path.write_text(json.dumps({"state": "aborted", "duration_s": args.duration,
                                           "seeds": args.seeds, "error": repr(error)}, indent=2), encoding="utf-8")
        raise
    status_path.write_text(json.dumps({"state": "complete", "duration_s": args.duration,
                                       "seeds": args.seeds, "snapshot": str(snapshot)}, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "complete", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
