#!/usr/bin/env python3
"""Offline delay-threshold sensitivity with explicit pending-packet bounds."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT

SEEDS = (41, 42)
THRESHOLDS = (20, 30, 40, 50)
RAW = "results/raw/tgax_fixed_actions_stepwise_seed{seed}_20s_20261025/A_default_beb"
OUT = ROOT / "results/summary/tgax_delay_threshold_sensitivity_20261025"


def read_success_delays(path: Path) -> dict[str, list[float]]:
    values = {"vo": [], "vi": []}
    with (path / "events.log").open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            fields = line.split()
            if not fields or fields[0] != "Y":
                continue
            time_s, ac = float(fields[1]), int(fields[3])
            # Raw control time: [0,1] is the common Default traffic warmup.
            if 1.0 < time_s <= 21.0 and ac in (0, 1):
                values[("vo", "vi")[ac]].append(float(fields[4]))
    return values


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in SEEDS:
        path = ROOT / RAW.format(seed=seed)
        summary = json.loads((path / "fixed_action_summary.json").read_text(encoding="utf-8"))
        delays = read_success_delays(path)
        for ac in ("vo", "vi"):
            success = sum(int(summary[f"{d}_{ac}_successful_samples"]) for d in ("ul", "dl"))
            risk = sum(int(summary[f"{d}_{ac}_risk_packets"]) for d in ("ul", "dl"))
            pending = sum(int(summary[f"{d}_{ac}_pending_end"]) for d in ("ul", "dl"))
            failures = 0
            for direction in ("ul", "dl"):
                failures += round(float(summary[f"{direction}_{ac}_final_drop_rate"])
                                  * int(summary[f"{direction}_{ac}_risk_packets"]))
            resolved = success + failures
            if success != len(delays[ac]) or resolved + pending != risk:
                raise RuntimeError(f"Ledger mismatch seed={seed} ac={ac}")
            for threshold in THRESHOLDS:
                timely = sum(value <= threshold for value in delays[ac])
                # Unknown end-of-simulation pending ages make the exact denominator
                # lie between all-risk and resolved-only denominators.
                lower = timely / risk if risk else None
                upper = timely / resolved if resolved else None
                native_threshold = (ac == "vo" and threshold == 20) or (ac == "vi" and threshold == 50)
                rows.append({
                    "seed": seed, "ac": ac.upper(), "threshold_ms": threshold,
                    "timely_successes": timely, "successful_mpdus": success,
                    "final_failures": failures, "pending_at_end": pending,
                    "risk_packets": risk, "resolved_packets": resolved,
                    "timely_rate_lower_bound": lower, "timely_rate_upper_bound": upper,
                    "bound_width": upper - lower if lower is not None else None,
                    "backend_exact_rate": summary[f"{ac}_on_time_delivery_rate"] if native_threshold else "",
                    "backend_exact_eligible": summary[f"{ac}_deadline_eligible"] if native_threshold else "",
                })
    with (OUT / "threshold_sensitivity.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    metadata = {
        "source": [RAW.format(seed=s) for s in SEEDS],
        "measurement_control_time": "(1,21] seconds; common 1 s Default warmup excluded",
        "definition": "timely successful MPDUs / eligible packets",
        "bounds": "pending packet ages are unavailable offline; lower includes all pending, upper excludes all pending",
        "exact_native_thresholds": {"VO_ms": 20, "VI_ms": 50},
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

