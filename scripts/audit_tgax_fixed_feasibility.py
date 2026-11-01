#!/usr/bin/env python3
"""Freeze the fixed-action subset and prove structural reachability of all 865 actions."""
from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config
from sim.edca.cw_growth import Growth

OUT = ROOT / "results/summary/tgax_fixed_feasibility_preflight_20261028"
ACS = ("VO", "VI", "BE", "BK")


def native(cwmin: int, cwmax: int) -> list[int]:
    return [cwmin] + [min(cwmax, (cwmin + 1) * 2**n - 1) for n in range(1, 11)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "configs/tgax_ddqn_reward_v3.yaml")
    definitions = json.loads((ROOT / "configs/tgax_fixed_feasibility_candidates.json").read_text(encoding="utf-8"))
    profiles = json.loads((ROOT / config["action"]["profiles"]).read_text(encoding="utf-8"))
    action = config["action"]
    scalar = list(itertools.product(range(len(action["alpha"])), range(len(action["beta1"])),
                                    range(len(action["beta2"])), range(len(action["gamma"]))))
    legal = [indices for indices in scalar if action["beta2"][indices[2]] >= action["beta1"][indices[1]]]
    executed = [(profile, *indices) for profile in range(16) for indices in legal]
    executed.append((action["default_fallback_index"], 0, 0, 0, 0))
    if len(executed) != 865 or len(set(executed)) != 865:
        raise RuntimeError("Reduced action enumeration is not exactly 865 unique actions")

    rows = []
    for name, spec in definitions["candidates"].items():
        chosen = spec["action"]; fallback = chosen[0] == action["default_fallback_index"]
        profile_index = action["default_profile"] if fallback else chosen[0]
        params = None if not spec["custom"] or fallback else (
            action["alpha"][chosen[1]], action["beta1"][chosen[2]],
            action["beta2"][chosen[3]], action["gamma"][chosen[4]])
        for ac in ACS:
            p = profiles[profile_index][ac]
            trajectory = native(p["CWmin"], p["CWmax"]) if params is None else Growth(*params).table(
                p["CWmin"], p["CWmax"], 10)
            rows.append({"candidate": name, "class": spec["class"], "description": spec["description"],
                         "requested_action": json.dumps(chosen), "executed_profile": profile_index,
                         "native_beb": params is None, "alpha": "" if params is None else params[0],
                         "beta1": "" if params is None else params[1], "beta2": "" if params is None else params[2],
                         "gamma": "" if params is None else params[3], "ac": ac,
                         "aifsn": p["AIFSN"], "cwmin": p["CWmin"], "cwmax": p["CWmax"],
                         "txop_ms": p["TXOP_ms"], "cw_trajectory_retry0_to10": " ".join(map(str, trajectory))})
    with (OUT / "candidate_parameters.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    prior = ROOT / "results/raw/tgax_full40_multiac_v6_action_smoke_20261026/reduced_action_audit.json"
    prior_audit = json.loads(prior.read_text(encoding="utf-8"))
    result = {"structural_action_count": len(executed), "profile_count": 16,
              "legal_scalar_tuples_per_profile": len(legal), "default_fallback_count": 1,
              "all_action_indices_unique_and_decodable": True,
              "candidate_count": len(definitions["candidates"]),
              "default_fallback_index": action["default_fallback_index"],
              "prior_runtime_control_audit": str(prior.relative_to(ROOT)),
              "prior_runtime_control_status": prior_audit["status"],
              "limitations": "Structural reachability covers all 865 actions; runtime application is sampled by prior smoke and current representative runs, not 865 exhaustive ns-3 executions."}
    (OUT / "action_space_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
