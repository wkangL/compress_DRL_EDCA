"""Validate corrected persistent Default against the one-shot baseline."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config


CAL = ROOT / "results/raw/mac_arrival40_gaming160_cmdp_default_calibration_20261029/calibration.json"
OLD = ROOT / "results/summary/mac_arrival40_gaming160_default_20s_3seed_20261029/per_seed_metrics.csv"
OUT = ROOT / "results/summary/mac_arrival40_gaming160_cmdp_gates_20261029"


def main() -> None:
    calibration = json.loads(CAL.read_text(encoding="utf-8"))
    old = {int(row["seed"]): row for row in csv.DictReader(OLD.open(encoding="utf-8-sig"))}
    rows = []
    for current in calibration["per_seed"]:
        seed = int(current["seed"])
        new_mbps = sum(current["acked_bytes"]) * 8 / current["duration_s"] / 1e6
        old_mbps = float(old[seed]["total_throughput_mbps"])
        rows.append({"seed": seed, "one_shot_mbps": old_mbps, "persistent_step_mbps": new_mbps,
                     "relative_difference": abs(new_mbps / old_mbps - 1),
                     "finite": current["all_finite"]})
    config = load_config(ROOT / "configs/mac_arrival40_gaming160_cmdp_calibrated.yaml")
    checks = {"all_three_seeds": len(rows) == 3,
              "throughput_within_one_percent": all(row["relative_difference"] < 0.01 for row in rows),
              "all_finite": all(row["finite"] for row in rows),
              "budgets_frozen": len(config["reward"]["budgets"]) == 4,
              "vo_vi_priority_weights": config["reward"]["utility_weights"] == [0.4, 0.4, 0.1, 0.1]}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "comparisons": rows,
              "budgets": config["reward"]["budgets"],
              "low_priority_min_satisfaction": config["reward"]["low_priority_min_satisfaction"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "default_equivalence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS": raise RuntimeError(result)


if __name__ == "__main__": main()
