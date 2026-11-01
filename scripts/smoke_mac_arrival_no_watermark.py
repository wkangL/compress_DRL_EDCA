#!/usr/bin/env python3
"""One-step correctness gate for immediate native WifiMacQueue admission."""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config
from sim.backend.base_backend import ControlAction
from sim.backend.tgax_ns3_backend import TgaxNs3Backend

OUTPUT = ROOT / "results/raw/mac_arrival40_gaming160_no_watermark_smoke_v2_20261030"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/mac_arrival40_gaming160_cmdp_no_watermark.yaml")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--duration", type=float, default=1.0)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    if output.exists():
        raise FileExistsError(f"Preserving existing smoke output: {output}")
    config = load_config(config_path)
    backend = TgaxNs3Backend(config, args.seed, output, args.duration)
    try:
        events = backend.step(ControlAction(0, False, 1, 2.0, 2.0, 7), args.duration)
    finally:
        backend.close()
    groups: dict[str, list[tuple[float, ...]]] = {}
    for event in events:
        groups.setdefault(event.kind, []).append(event.values)
    demand = groups.get("U", [])
    ledgers = groups.get("M", [])
    checks = {
        "requested_duration_completed": abs(backend.time - args.duration) < 1e-9,
        "eight_direction_ac_ledgers": len(ledgers) == 8,
        "four_ac_demand_ledgers": len(demand) == 4,
        "forty_sta_demand_ledgers": len(groups.get("V", [])) == 40,
        "all_generated_bytes_immediately_offered_to_mac": all(row[1] == row[3] for row in demand),
        # M: direction, AC, arrived, acked, failed, pending_start,
        # pending_end, bytes..., attempts, retries, eligible, on_time.
        "mac_ledger_conserved": all(row[5] + row[2] == row[3] + row[4] + row[6] for row in ledgers),
        "deadline_ledger_causal": all(row[13] <= row[12] for row in ledgers),
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "event_counts": {key: len(value) for key, value in sorted(groups.items())},
              "ac_demand_ledgers": demand, "mac_ledgers": ledgers}
    (output / "smoke_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise RuntimeError(result)


if __name__ == "__main__":
    main()
