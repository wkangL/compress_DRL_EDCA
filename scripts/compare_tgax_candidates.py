"""Compare the original and increased real-time-concurrency candidates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "p95_final_mac_service_delay_ms",
    "successful_mac_throughput_mbps",
    "final_mac_drop_rate",
    "retransmission_rate",
)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    old = {(r["ac"], r["direction"]): r for r in read(args.old / "bss0_ac_direction_metrics.csv")}
    new = {(r["ac"], r["direction"]): r for r in read(args.new / "bss0_ac_direction_metrics.csv")}
    rows = []
    for key in sorted(set(old) | set(new)):
        for field in FIELDS:
            old_raw, new_raw = old[key].get(field, ""), new[key].get(field, "")
            old_value = float(old_raw) if old_raw not in ("", None) else None
            new_value = float(new_raw) if new_raw not in ("", None) else None
            delta = new_value - old_value if old_value is not None and new_value is not None else None
            percent = delta / old_value * 100 if delta is not None and old_value else None
            rows.append({"ac": key[0], "direction": key[1], "metric": field,
                         "old": "" if old_value is None else old_value,
                         "new": "" if new_value is None else new_value,
                         "absolute_change": "" if delta is None else delta,
                         "relative_change_percent": "" if percent is None else percent})
    with (args.output / "old_vs_new_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    old_v = json.loads((args.old / "candidate_validation.json").read_text(encoding="utf-8"))
    new_v = json.loads((args.new / "candidate_validation.json").read_text(encoding="utf-8"))
    summary = {
        "old_active_business_stas": old_v["active_business_sta_count"],
        "new_active_business_stas": new_v["active_business_sta_count"],
        "old_jain_demand_satisfaction": old_v["jain_demand_satisfaction"],
        "new_jain_demand_satisfaction": new_v["jain_demand_satisfaction"],
        "old_validation_status": old_v["status"], "new_validation_status": new_v["status"],
    }
    (args.output / "old_vs_new_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
