#!/usr/bin/env python3
"""Finish final-only held-out evaluation after a completed training snapshot.

This recovery entry point deliberately performs no gradient update and no
checkpoint selection.  It is used when all 600 training steps completed but a
post-training output-directory bug prevented evaluation from starting.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config
from scripts.run_tgax_ddqn_extended import (
    action_histogram, evaluate_one, load_snapshot_model, save_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    root = args.root if args.root.is_absolute() else ROOT / args.root
    config = load_config(config_path)
    expected_steps = int(config["training"]["stages"] * config["training"]["episode_s"]
                         / config["timing"]["decision_s"])
    final_snapshot = root / "snapshots" / f"step_{expected_steps:03d}.pt"
    if not final_snapshot.is_file():
        raise FileNotFoundError(final_snapshot)
    (root / "status.json").write_text(json.dumps({
        "state": "final_evaluation", "steps": expected_steps,
        "held_out_seeds": config["evaluation"]["held_out_seeds"],
    }, indent=2), encoding="utf-8")

    validation = root / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    selection = {
        "metric": "not applicable: user required final checkpoint only",
        "final_only_evaluation": True,
        "selected_step": expected_steps,
        "selected_snapshot": str(final_snapshot.relative_to(root)),
        "final_snapshot": str(final_snapshot.relative_to(root)),
    }
    (validation / "checkpoint_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8")

    model, scaler = load_snapshot_model(final_snapshot, config)
    rows = []
    for seed in config["evaluation"]["held_out_seeds"]:
        default, _ = evaluate_one(config, model, scaler, seed,
                                  root / "held_out" / f"seed_{seed}" / "default", False)
        final, _ = evaluate_one(config, model, scaler, seed,
                                root / "held_out" / f"seed_{seed}" / "final", True)
        base_reward = float(default["reward_mean"])
        rows.append({
            "eval_seed": seed, "policy": "final", "checkpoint_step": expected_steps,
            "default_reward": base_reward, "learned_reward": final["reward_mean"],
            "reward_change_percent": ((float(final["reward_mean"]) / base_reward - 1) * 100
                                      if base_reward else None),
            "paired_generated_demand_equal_each_step":
                default["application_generated_by_step"] == final["application_generated_by_step"],
        })
    save_csv(root / "held_out" / "comparison.csv", rows)

    training_rows = []
    with (root / "training_steps.csv").open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            training_rows.append({"action": json.loads(row["action"])})
    (root / "action_distribution.json").write_text(
        json.dumps(action_histogram(training_rows), indent=2), encoding="utf-8")
    checkpoint = root / "train" / "checkpoint.pt"
    losses_path = root / "losses.csv"
    updates = 0
    if losses_path.is_file():
        with losses_path.open(newline="", encoding="utf-8-sig") as stream:
            updates = sum(1 for _ in csv.DictReader(stream))
    saved_status = {
        "state": "complete", "steps": expected_steps,
        "updates": updates, "selected_step": expected_steps,
        "held_out_seeds": config["evaluation"]["held_out_seeds"],
        "recovered_post_training_finalization": True,
        "checkpoint": str(checkpoint),
    }
    (root / "status.json").write_text(json.dumps(saved_status, indent=2), encoding="utf-8")
    print(json.dumps(saved_status), flush=True)


if __name__ == "__main__":
    main()
