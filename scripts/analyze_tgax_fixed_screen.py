#!/usr/bin/env python3
"""Summarize the predeclared 5 s screen and freeze formal candidates."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT

RUN = ROOT / "results/raw/tgax_fixed_feasibility_screen_5s_2seed_20261028"
OUT = ROOT / "results/summary/tgax_fixed_feasibility_screen_20261028"
ACS = ("vo", "vi", "be", "bk")


def aggregate_p95(path: Path, ac_index: int, duration: float) -> float | None:
    samples = []
    with (path / "events.log").open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            f = line.split()
            if f and f[0] == "Y" and 1.0 < float(f[1]) <= 1.0 + duration and int(float(f[3])) == ac_index:
                samples.append(float(f[4]))
    return float(np.percentile(samples, 95)) if samples else None


def pareto(rows: list[dict]) -> set[str]:
    front = set()
    for current in rows:
        dominated = False
        for other in rows:
            if other is current: continue
            no_worse = (other["application_throughput_mbps"] >= current["application_throughput_mbps"] and
                        other["combined_delay_normalized"] <= current["combined_delay_normalized"] and
                        other["jain"] >= current["jain"])
            strict = (other["application_throughput_mbps"] > current["application_throughput_mbps"] or
                      other["combined_delay_normalized"] < current["combined_delay_normalized"] or
                      other["jain"] > current["jain"])
            if no_worse and strict: dominated = True; break
        if not dominated: front.add(current["candidate"])
    return front


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    experiment = json.loads((RUN / "experiment_config.json").read_text(encoding="utf-8"))
    duration = float(experiment["duration_s"]); seeds = experiment["seeds"]
    per_seed = []
    for seed in seeds:
        for name, spec in experiment["candidates"].items():
            path = RUN / f"seed_{seed}" / name
            summary = json.loads((path / "fixed_action_summary.json").read_text(encoding="utf-8"))
            generated = np.asarray(summary["application_generated_bytes"], dtype=float)
            submitted = np.asarray(summary["application_submitted_bytes"], dtype=float)
            received = np.asarray(summary["application_received_bytes"], dtype=float)
            row = {"seed": seed, "candidate": name, "class": spec["class"],
                   "action": json.dumps(spec["action"]), "custom": spec["custom"],
                   "application_throughput_mbps": float(received.sum() * 8 / duration / 1e6),
                   "application_satisfaction": float(received.sum() / generated.sum()),
                   "socket_submission_fraction": float(submitted.sum() / generated.sum()),
                   "jain": summary["jain_demand_satisfaction"],
                   "vo_p95_ms": aggregate_p95(path, 0, duration),
                   "vi_p95_ms": aggregate_p95(path, 1, duration),
                   "vo_timely": summary["vo_on_time_delivery_rate"],
                   "vi_timely": summary["vi_on_time_delivery_rate"]}
            for i, ac in enumerate(ACS):
                row[f"{ac}_application_throughput_mbps"] = float(received[i] * 8 / duration / 1e6)
                row[f"{ac}_application_satisfaction"] = float(received[i] / generated[i]) if generated[i] else None
            row["combined_delay_normalized"] = (0.5 * (row["vo_p95_ms"] / 20.0 + row["vi_p95_ms"] / 50.0)
                                                if row["vo_p95_ms"] is not None and row["vi_p95_ms"] is not None
                                                else None)
            per_seed.append(row)
    keys = list(per_seed[0])
    with (OUT / "screen_per_seed.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(per_seed)
    mean_rows = []
    for name in experiment["candidates"]:
        group = [r for r in per_seed if r["candidate"] == name]
        row = {"candidate": name, "class": group[0]["class"], "action": group[0]["action"],
               "custom": group[0]["custom"]}
        for key in keys:
            if key in {"seed", "candidate", "class", "action", "custom"}: continue
            values = [r[key] for r in group if r[key] is not None]
            row[key] = float(np.mean(values)) if values else None
        row["vo_p95_valid_seed_count"] = sum(r["vo_p95_ms"] is not None for r in group)
        row["vi_p95_valid_seed_count"] = sum(r["vi_p95_ms"] is not None for r in group)
        mean_rows.append(row)
    front = pareto([r for r in mean_rows if r["combined_delay_normalized"] is not None and
                    r["jain"] is not None])
    default = next(r for r in mean_rows if r["candidate"] == "default_native")
    for row in mean_rows:
        row["pareto"] = row["candidate"] in front
        row["throughput_vs_default_percent"] = (row["application_throughput_mbps"] / default["application_throughput_mbps"] - 1) * 100
        row["combined_delay_vs_default_percent"] = ((row["combined_delay_normalized"] / default["combined_delay_normalized"] - 1) * 100
                                                     if row["combined_delay_normalized"] is not None else None)
        row["jain_vs_default_percent"] = ((row["jain"] / default["jain"] - 1) * 100
                                           if row["jain"] is not None else None)
        row["severe_guard"] = (row["application_throughput_mbps"] >= 0.2 * default["application_throughput_mbps"] and
                               row["vo_timely"] >= 0.5 * default["vo_timely"] and
                               row["vi_timely"] >= 0.5 * default["vi_timely"] and
                               row["combined_delay_normalized"] is not None)

    eligible = [r for r in mean_rows if r["severe_guard"]]
    throughput_choice = max(eligible, key=lambda r: r["application_throughput_mbps"])["candidate"]
    custom_joint = [r for r in eligible if r["custom"] and not r["candidate"].startswith("p0_") and
                    r["application_throughput_mbps"] >= default["application_throughput_mbps"] and
                    r["combined_delay_normalized"] < default["combined_delay_normalized"] and
                    r["jain"] >= 0.75 * default["jain"]]
    joint_choice = min(custom_joint, key=lambda r: r["combined_delay_normalized"])["candidate"]
    p0_safe = [r for r in eligible if r["candidate"].startswith("p0_c_") and
               r["vo_timely"] >= 0.9 * default["vo_timely"] and r["vi_timely"] >= 0.95 * default["vi_timely"]]
    be_choice = max(p0_safe, key=lambda r: r["be_application_throughput_mbps"] + r["bk_application_throughput_mbps"])["candidate"]
    latency_choice = min([r for r in p0_safe if r["candidate"] != be_choice],
                         key=lambda r: r["combined_delay_normalized"])["candidate"]
    selected = ["default_native", throughput_choice, joint_choice, be_choice, latency_choice]
    if len(set(selected)) != 5: raise RuntimeError(f"Selection roles are not distinct: {selected}")
    reasons = {"default_native": "paired native baseline", throughput_choice: "highest screened application throughput",
               joint_choice: "custom non-P0 action with throughput>=Default, lower combined delay and Jain>=75% Default",
               be_choice: "P0 CW-only action with highest BE+BK payload throughput under VO/VI timely guards",
               latency_choice: "remaining P0 CW-only action with lowest combined delay under VO/VI timely guards"}
    for row in mean_rows: row["formal_selected"] = row["candidate"] in selected; row["selection_reason"] = reasons.get(row["candidate"], "")
    with (OUT / "screen_mean.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mean_rows[0])); writer.writeheader(); writer.writerows(mean_rows)
    formal = {"description": "Frozen formal candidates selected by predeclared multi-objective screen roles",
              "combined_delay_formula": "0.5*(VO aggregate p95/20ms + VI aggregate p95/50ms)",
              "selection_rules": reasons, "candidates": {name: experiment["candidates"][name] for name in selected}}
    (OUT / "formal_candidates.json").write_text(json.dumps(formal, indent=2), encoding="utf-8")
    (OUT / "screen_selection.json").write_text(json.dumps({"pareto": sorted(front), "selected": selected,
        "rules_fixed_before_formal_seed44_45_runs": True, "formal_test_seeds": [44, 45]}, indent=2), encoding="utf-8")
    print(json.dumps({"pareto": sorted(front), "selected": selected, "reasons": reasons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
