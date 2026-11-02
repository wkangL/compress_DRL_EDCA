#!/usr/bin/env python3
"""Reproducible synthetic gate for aggregate versus directional CMDP costs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.cmdp_reward import cmdp_components, directional_cmdp_components, lagrangian_reward


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/summary/cmdp_directional_reward_fastcheck_20261101"
CONFIG = {
    "utility_weights": [0.4, 0.4, 0.1, 0.1],
    "delay_reference_ms": [20.0, 50.0],
    "delay_excess_cap": 3.0,
    "minimum_jain": 0.5,
    "minimum_satisfaction": [0.02, 0.005],
}
OLD_BUDGETS = np.asarray([0.4, 0.08, 0.0, 0.3])
OLD_LAMBDA = np.asarray([0.1, 0.15, 0.1, 0.2])
NEW_BUDGETS = np.asarray([0.4, 0.4, 0.08, 0.08, 0.0, 0.3])
NEW_LAMBDA = np.asarray([0.05, 0.05, 0.075, 0.075, 0.1, 0.2])


def baseline() -> dict:
    metrics = {
        "demand_satisfaction": [0.7, 0.5, 0.02, 0.005],
        "demand_active": [True, True, True, True],
        "jain_demand_satisfaction": 0.6,
        "jain_active_sta_count": 40,
    }
    for ac in ("vo", "vi"):
        for direction in ("ul", "dl"):
            prefix = f"{direction}_{ac}"
            metrics.update({f"{prefix}_pending_start": 0, f"{prefix}_arrived": 10,
                            f"{prefix}_acked": 10, f"{prefix}_failed": 0,
                            f"{prefix}_pending_end": 0,
                            f"{prefix}_delay_samples_ms": [10.0] * 10})
    return metrics


def scenario(name: str) -> dict:
    metrics = baseline()
    if name == "healthy":
        return metrics
    if name == "ul_vo_delay_bad":
        metrics["ul_vo_delay_samples_ms"] = [80.0] * 10
        return metrics
    if name == "dl_vi_no_success":
        metrics.update(dl_vi_acked=0, dl_vi_failed=0, dl_vi_pending_end=10,
                       dl_vi_delay_samples_ms=[])
        return metrics
    if name == "ul_vo_final_loss":
        metrics.update(ul_vo_acked=8, ul_vo_failed=2,
                       ul_vo_delay_samples_ms=[10.0] * 8)
        return metrics
    raise KeyError(name)


def main() -> None:
    rows = []
    for name in ("healthy", "ul_vo_delay_bad", "dl_vi_no_success", "ul_vo_final_loss"):
        metrics = scenario(name)
        utility, old_costs, old_parts = cmdp_components(metrics, CONFIG)
        _, new_costs, new_parts = directional_cmdp_components(metrics, CONFIG)
        rows.append({
            "scenario": name,
            "utility": utility,
            "aggregate_costs": old_costs.tolist(),
            "directional_costs": new_costs.tolist(),
            "aggregate_budget_violation": (old_costs > OLD_BUDGETS).tolist(),
            "directional_budget_violation": (new_costs > NEW_BUDGETS).tolist(),
            "aggregate_reward": lagrangian_reward(utility, old_costs, OLD_BUDGETS, OLD_LAMBDA),
            "directional_reward": lagrangian_reward(utility, new_costs, NEW_BUDGETS, NEW_LAMBDA),
            "aggregate_order": old_parts["cost_order"],
            "directional_order": new_parts["cost_order"],
        })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "validation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (OUTPUT / "validation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                             for key, value in row.items()})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
