#!/usr/bin/env python3
"""Re-score completed v6 step-200 validation logs with Reward-v3.

Historical logs contain exact per-window deadline and per-AC demand ledgers, but
not per-window per-STA ledgers.  The final flow-summary Jain value is therefore
used only as a clearly labelled episode-level surrogate in this retrospective
check.  New Reward-v3 runs record exact online per-STA Jain values.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT


OUT = ROOT / "results/summary/tgax_reward_v3_offline_20261027"
OLD = "tgax_ddqn_full40_multiac_v6_seed{}_400step_20261027"


def load_steps(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row["application_satisfaction"] = json.loads(row["application_satisfaction"])
            row["application_demand_active"] = json.loads(row["application_demand_active"])
            rows.append(row)
    return rows


def score(directory: Path) -> dict:
    summary = json.loads((directory / "evaluation_summary.json").read_text(encoding="utf-8"))
    rows = load_steps(directory / "evaluation_steps.csv")
    jain = float(summary["jain_demand_satisfaction"])
    rewards = []
    services = []
    for row in rows:
        active = np.asarray(row["application_demand_active"], dtype=bool)
        satisfaction = np.asarray(row["application_satisfaction"], dtype=float)
        service = float(satisfaction[active].mean()) if active.any() else 0.0
        vo = float(row["vo_on_time_delivery_rate"]) if row["vo_on_time_delivery_valid"] == "True" else 0.0
        vi = float(row["vi_on_time_delivery_rate"]) if row["vi_on_time_delivery_valid"] == "True" else 0.0
        services.append(service)
        rewards.append(0.4 * vo + 0.4 * vi + 0.1 * service + 0.1 * jain)
    return {
        "old_reward_mean": float(summary["reward_mean"]),
        "reward_v3_offline_mean": float(np.mean(rewards)),
        "vo_timely_aggregate": summary["vo_on_time_delivery_rate"],
        "vi_timely_aggregate": summary["vi_on_time_delivery_rate"],
        "service_all_window_mean": float(np.mean(services)),
        "jain_episode_surrogate": jain,
        "jain_online_exact": False,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for train_seed in (41, 42, 43):
        root = ROOT / "results/raw" / OLD.format(train_seed) / "validation"
        for eval_seed in (44, 45):
            for policy, relative in (("default", Path("baseline") / f"seed_{eval_seed}"),
                                     ("old_ddqn_step200", Path("step_200") / f"seed_{eval_seed}")):
                records.append({"train_seed": train_seed, "eval_seed": eval_seed,
                                "policy": policy, **score(root / relative)})
    path = OUT / "reward_v3_offline_validation.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    checks = []
    for train_seed in (41, 42, 43):
        for eval_seed in (44, 45):
            pair = [r for r in records if r["train_seed"] == train_seed and r["eval_seed"] == eval_seed]
            default = next(r for r in pair if r["policy"] == "default")
            learned = next(r for r in pair if r["policy"] != "default")
            checks.append({"train_seed": train_seed, "eval_seed": eval_seed,
                           "default_reward_v3": default["reward_v3_offline_mean"],
                           "old_ddqn_reward_v3": learned["reward_v3_offline_mean"],
                           "default_higher": default["reward_v3_offline_mean"] > learned["reward_v3_offline_mean"]})
    (OUT / "offline_check.json").write_text(json.dumps({
        "method": "exact historical deadline/per-AC service ledgers plus final episode Jain surrogate",
        "limitation": "historical v6 lacks per-window per-STA demand/delivery; new v3 protocol adds exact V records",
        "all_pairs_pass": all(x["default_higher"] for x in checks),
        "pairs": checks,
    }, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
