#!/usr/bin/env python3
"""Build an exact AC-aggregated comparison from completed validation runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ACS = ("vo", "vi", "be", "bk")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def summarize(run: Path) -> dict:
    summary = json.loads((run / "evaluation_summary.json").read_text(encoding="utf-8"))
    steps = read_csv(run / "evaluation_steps.csv")
    delays = {ac: [] for ac in ACS}
    with (run / "events.log").open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if fields and fields[0] == "Y":
                delays[ACS[int(float(fields[3]))]].append(float(fields[4]))
    result = {
        "reward": float(summary["reward_mean"]),
        "jain": summary["jain_demand_satisfaction"],
        "vo_timely": summary["vo_on_time_delivery_rate"],
        "vi_timely": summary["vi_on_time_delivery_rate"],
    }
    for ac in ACS:
        expected = sum(int(summary[f"{direction}_{ac}_successful_samples"])
                       for direction in ("ul", "dl"))
        # events.log also contains the one-second common Default traffic warmup.
        # The evaluation summary excludes it, so retain exactly the final N Y
        # records belonging to the 20 decision windows.
        selected = delays[ac][-expected:] if expected else []
        result[f"{ac}_p95_ms"] = float(np.percentile(selected, 95)) if selected else None
        result[f"{ac}_samples"] = len(selected)
        result[f"{ac}_throughput_mbps"] = sum(
            float(summary[f"{direction}_{ac}_throughput_mbps"] or 0) for direction in ("ul", "dl"))
        risk = sum(int(summary[f"{direction}_{ac}_risk_packets"]) for direction in ("ul", "dl"))
        failed = sum(int(round(int(summary[f"{direction}_{ac}_risk_packets"]) *
                               float(summary[f"{direction}_{ac}_final_drop_rate"] or 0)))
                     for direction in ("ul", "dl"))
        attempts = sum(int(float(row[f"{direction}_{ac}_tx_attempts"]))
                       for row in steps for direction in ("ul", "dl"))
        retries = sum(int(float(row[f"{direction}_{ac}_retransmissions"]))
                      for row in steps for direction in ("ul", "dl"))
        result[f"{ac}_failure_rate"] = failed / risk if risk else None
        result[f"{ac}_retransmission_rate"] = retries / attempts if attempts else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for root in args.roots:
        train_seed = int(root.name.split("seed", 1)[1].split("_", 1)[0])
        for eval_seed in (44, 45):
            for method, run in (
                ("Default", root / "validation" / "baseline" / f"seed_{eval_seed}"),
                ("DDQN@200", root / "validation" / "step_200" / f"seed_{eval_seed}"),
            ):
                if (run / "evaluation_summary.json").is_file():
                    records.append({"train_seed": train_seed, "eval_seed": eval_seed,
                                    "method": method, **summarize(run)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if records:
        with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    print(json.dumps(records, ensure_ascii=False))


if __name__ == "__main__":
    main()
