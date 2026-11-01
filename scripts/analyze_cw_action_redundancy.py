"""Enumerate exact CW trajectories produced by the ns-3 research growth hook."""
from __future__ import annotations

import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = json.loads((ROOT / "configs/profiles.json").read_text(encoding="utf-8"))
OUT = ROOT / "results/summary/tgax_cw_redundancy_20261025"
OUT.mkdir(parents=True, exist_ok=True)
ACS = ("VO", "VI", "BE", "BK")
ALPHAS = (1, 2, 3)
BETA1 = (10, 12, 14, 16, 18, 20)
BETA2 = tuple(range(10, 31, 2))
GAMMAS = (5, 6, 7)
RETRY_LIMIT = 10


def custom_trajectory(cwmin: int, cwmax: int, alpha: int, beta1: int, beta2: int, gamma: int) -> tuple[int, ...]:
    cw = cwmin
    values = [cw]
    for retry in range(1, RETRY_LIMIT + 1):
        if retry <= gamma:
            beta = beta1 if retry <= alpha else beta2
            cw = min(cwmax, (beta * (cw + 1) + 9) // 10 - 1)
        values.append(cw)
    return tuple(values)


def native_trajectory(cwmin: int, cwmax: int) -> tuple[int, ...]:
    return tuple([cwmin] + [min(cwmax, (1 << retry) * (cwmin + 1) - 1)
                             for retry in range(1, RETRY_LIMIT + 1)])


def legal_parameters():
    for alpha, beta1, beta2, gamma in itertools.product(ALPHAS, BETA1, BETA2, GAMMAS):
        if beta2 >= beta1:
            yield alpha, beta1, beta2, gamma


PARAMETERS = list(legal_parameters())
assert len(PARAMETERS) == 459


def no_effect_group_fraction(records: list[tuple[tuple[int, int, int, int], tuple]], parameter_index: int) -> tuple[int, int, float]:
    groups: dict[tuple, set[tuple]] = defaultdict(set)
    for params, trajectory in records:
        key = params[:parameter_index] + params[parameter_index + 1:]
        groups[key].add(trajectory)
    no_effect = sum(len(values) == 1 for values in groups.values())
    return no_effect, len(groups), no_effect / len(groups)


stats = []
representatives = []
all_trajectories = []
for profile_index, profile in enumerate(PROFILES):
    records = []
    exact: dict[tuple, list[tuple[int, int, int, int]]] = defaultdict(list)
    per_ac = [set() for _ in ACS]
    first3 = set()
    saturation = {ac: {1: 0, 2: 0, 3: 0} for ac in ACS}
    native = tuple(native_trajectory(profile[ac]["CWmin"], profile[ac]["CWmax"]) for ac in ACS)
    native_equivalent = 0
    for params in PARAMETERS:
        alpha, beta1, beta2, gamma = params
        trajectories = tuple(custom_trajectory(profile[ac]["CWmin"], profile[ac]["CWmax"],
                                                alpha, beta1, beta2, gamma) for ac in ACS)
        records.append((params, trajectories)); exact[trajectories].append(params)
        first3.add(tuple(t[:4] for t in trajectories))
        native_equivalent += trajectories == native
        for index, (ac, trajectory) in enumerate(zip(ACS, trajectories)):
            per_ac[index].add(trajectory)
            for retry in (1, 2, 3):
                saturation[ac][retry] += trajectory[retry] == profile[ac]["CWmax"]
        all_trajectories.append({"profile": profile_index, "alpha": alpha, "beta1": beta1 / 10,
                                 "beta2": beta2 / 10, "gamma": gamma,
                                 **{f"{ac}_trajectory": " ".join(map(str, trajectories[i])) for i, ac in enumerate(ACS)}})
    beta2_no, beta2_groups, beta2_fraction = no_effect_group_fraction(records, 2)
    gamma_no, gamma_groups, gamma_fraction = no_effect_group_fraction(records, 3)
    alpha_no, alpha_groups, alpha_fraction = no_effect_group_fraction(records, 0)
    row = {"profile": profile_index, "legal_actions": len(PARAMETERS),
           "exact_joint_trajectories": len(exact),
           "exact_duplicate_actions": len(PARAMETERS) - len(exact),
           "exact_duplicate_ratio": 1 - len(exact) / len(PARAMETERS),
           "first3_joint_trajectories_approx": len(first3),
           "first3_merge_ratio_approx": 1 - len(first3) / len(PARAMETERS),
           "native_beb_equivalent_custom_actions": native_equivalent,
           "alpha_no_effect_groups": alpha_no, "alpha_groups": alpha_groups,
           "alpha_no_effect_group_ratio": alpha_fraction,
           "beta2_no_effect_groups": beta2_no, "beta2_groups": beta2_groups,
           "beta2_no_effect_group_ratio": beta2_fraction,
           "gamma_no_effect_groups": gamma_no, "gamma_groups": gamma_groups,
           "gamma_no_effect_group_ratio": gamma_fraction}
    for index, ac in enumerate(ACS):
        row[f"{ac}_exact_trajectories"] = len(per_ac[index])
        for retry in (1, 2, 3):
            row[f"{ac}_at_cwmax_by_retry{retry}_ratio"] = saturation[ac][retry] / len(PARAMETERS)
    stats.append(row)
    for trajectory, params in exact.items():
        canonical = min(params)
        representatives.append({"profile": profile_index, "alpha": canonical[0],
                                "beta1": canonical[1] / 10, "beta2": canonical[2] / 10,
                                "gamma": canonical[3], "equivalent_action_count": len(params),
                                **{f"{ac}_trajectory": " ".join(map(str, trajectory[i])) for i, ac in enumerate(ACS)}})


def write_csv(path: Path, data: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)


write_csv(OUT / "profile_redundancy.csv", stats)
write_csv(OUT / "exact_representatives.csv", representatives)
write_csv(OUT / "all_action_trajectories.csv", all_trajectories)

selected = []
for profile_index in (0, 8):
    p = PROFILES[profile_index]
    for mode, params in (("native_BEB", None), ("custom_a2_b1.6_b2.8_g6", (2, 16, 28, 6))):
        item = {"profile": profile_index, "mode": mode}
        for ac in ACS:
            trajectory = native_trajectory(p[ac]["CWmin"], p[ac]["CWmax"]) if params is None else custom_trajectory(
                p[ac]["CWmin"], p[ac]["CWmax"], *params)
            item[f"{ac}_trajectory"] = " ".join(map(str, trajectory))
        selected.append(item)
write_csv(OUT / "selected_action_trajectories.csv", selected)

summary = {"retry_limit": RETRY_LIMIT, "legal_parameter_tuples_per_profile": len(PARAMETERS),
           "total_legal_actions": len(PARAMETERS) * len(PROFILES),
           "total_exact_joint_trajectories_within_profiles": sum(r["exact_joint_trajectories"] for r in stats),
           "exact_deduplication_ratio": 1 - sum(r["exact_joint_trajectories"] for r in stats) / (len(PARAMETERS) * len(PROFILES)),
           "total_first3_joint_trajectories_approx": sum(r["first3_joint_trajectories_approx"] for r in stats),
           "first3_merge_ratio_approx": 1 - sum(r["first3_joint_trajectories_approx"] for r in stats) / (len(PARAMETERS) * len(PROFILES)),
           "formula": "cw'=min(cwmax,ceil(beta*(cw+1))-1), beta1 for retry<=alpha else beta2, update only retry<=gamma",
           "profiles": stats}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "profiles"}, indent=2))
