#!/usr/bin/env python3
"""Two-STEP ns-3 smoke gate for the configured exact-QoS CMDP ledger."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config
from rl.tgax_environment import TgaxControlEnvironment
from scripts.run_mac_cmdp_experiment import paths, prepare, read_yaml, validate


SOURCE = ROOT / "configs/mac_cmdp_experiment.yaml"
OUTPUT = ROOT / "results/raw/mac_cmdp_five_cost_smoke_20261102"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Preserving existing smoke output: {OUTPUT}")
    spec = validate(read_yaml(SOURCE))
    generated = prepare(spec, SOURCE, paths(spec, OUTPUT))
    config = load_config(generated)
    action = np.asarray(config["action"].get("default_action", [0, 0, 5, 5, 2]), dtype=np.int64)
    env = TgaxControlEnvironment(config, "default", 9901, OUTPUT / "backend", 2.0, 0)
    records = []
    try:
        observation = env.reset_observation()
        assert observation["history"].shape == (40, 4, 10, 6)
        assert observation["global_stats"].shape == (10,)
        for step in range(2):
            observation, reward, metrics = env.step(action, 1.0, custom=False)
            costs = np.asarray(metrics["cmdp_costs"], dtype=float)
            assert costs.shape == (5,) and np.isfinite(costs).all()
            assert metrics["cmdp_cost_order"] == ["qos_vo_dl", "qos_vo_ul", "qos_vi_dl", "qos_vi_ul", "starvation"]
            assert metrics["cmdp_cost_valid"][:4] == [True, True, True, True]
            assert (costs[:4] > 0.0).all()
            records.append({
                "step": step + 1,
                "utility": metrics["cmdp_utility"],
                "qos_vo_dl_cost": costs[0],
                "qos_vo_ul_cost": costs[1],
                "qos_vi_dl_cost": costs[2],
                "qos_vi_ul_cost": costs[3],
                "starvation_cost": costs[4],
                "cost_valid": metrics["cmdp_cost_valid"],
                "qos_vo_ul": metrics["cmdp_qos_vo_ul"],
                "qos_vo_dl": metrics["cmdp_qos_vo_dl"],
                "qos_vi_ul": metrics["cmdp_qos_vi_ul"],
                "qos_vi_dl": metrics["cmdp_qos_vi_dl"],
                "be_satisfaction": metrics["cmdp_satisfaction_be"],
                "bk_satisfaction": metrics["cmdp_satisfaction_bk"],
                "reward": reward,
            })
    finally:
        env.close()
    summary = {
        "state_shapes": {"history": [40, 4, 10, 6], "global": [10]},
        "cost_order": config["reward"]["cost_order"],
        "budgets": config["reward"]["budgets"],
        "multipliers": config["reward"]["multiplier_initial"],
        "records": records,
    }
    (OUTPUT / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
