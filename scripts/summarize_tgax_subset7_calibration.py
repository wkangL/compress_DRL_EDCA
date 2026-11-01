"""Aggregate audited subset7 runs into analysis-ready CSV tables."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def find_runs(roots: list[Path]) -> list[Path]:
    found = []
    for root in roots:
        for validation in root.rglob("calibration_validation.json"):
            if json.loads(validation.read_text(encoding="utf-8")).get("status") == "PASS":
                found.append(validation.parent)
    return sorted(set(found))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = find_runs(args.root)
    run_rows: list[dict] = []
    ac_rows: list[dict] = []
    for run in runs:
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        load = json.loads((run / "load_manifest.json").read_text(encoding="utf-8"))
        bss = {int(row["bss"]): row for row in read_csv(run / "bss_summary.csv")}
        run_rows.append({
            "variant": manifest["variant"], "seed": manifest["seed"],
            "measurement_s": manifest["measurement_s"], "profile": manifest["profile"],
            "classification": load["classification"], "scale": load["scale"],
            "realtime_only": load["realtime_only"],
            "wall_s": manifest["elapsed_wall_seconds"], "peak_rss_mib": manifest.get("peak_rss_mib", ""),
            "target_throughput_mbps": bss[0]["throughput_mbps"],
            "target_delivery_fraction": bss[0]["delivery_fraction"],
            "target_busy_fraction": bss[0]["busy_fraction"],
            "target_mean_queue_packets": bss[0]["mean_queue_packets"],
            "target_max_queue_packets": bss[0]["max_queue_packets"],
            "neighbor_mean_busy_fraction": sum(float(bss[i]["busy_fraction"]) for i in bss if i != 0) / (len(bss) - 1),
            "run_path": str(run.resolve()),
        })
        app = defaultdict(lambda: [0, 0])
        for row in read_csv(run / "flow_summary.csv"):
            scope = "TARGET" if int(row["bss"]) == 0 else "NEIGHBORS"
            key = (scope, row["direction"], row["ac"])
            app[key][0] += int(row["offered_bytes"])
            app[key][1] += int(row["received_bytes"])
        outcomes = {(r["scope"], r["direction"], r["ac"]): r
                    for r in read_csv(run / "mac_qos_outcomes.csv") if r["scope"] in {"TARGET", "NEIGHBORS"}}
        delays = {(r["scope"], r["direction"], r["ac"]): r
                  for r in read_csv(run / "mac_service_delay_summary.csv") if r["scope"] in {"TARGET", "NEIGHBORS"}}
        keys = sorted(set(app) | set(outcomes) | set(delays))
        for scope, direction, ac in keys:
            offered, received = app[(scope, direction, ac)]
            outcome = outcomes.get((scope, direction, ac), {})
            delay = delays.get((scope, direction, ac), {})
            ac_rows.append({
                "variant": manifest["variant"], "seed": manifest["seed"],
                "measurement_s": manifest["measurement_s"], "scope": scope,
                "direction": direction, "ac": ac,
                "offered_bytes": offered, "received_bytes": received,
                "application_delivery_fraction": received / offered if offered else 0,
                "mac_arrived": outcome.get("arrived", 0), "mac_acked": outcome.get("acked", 0),
                "mac_failed": outcome.get("failed", 0), "mac_unfinished": outcome.get("unfinished", 0),
                "mac_tx_attempts": outcome.get("tx_attempts", 0),
                "mac_retransmissions": outcome.get("retransmissions", 0),
                "mac_ack_fraction": outcome.get("ack_fraction", 0),
                "mac_failure_fraction": outcome.get("failure_fraction", 0),
                "mac_unfinished_fraction": outcome.get("unfinished_fraction", 0),
                "successful_packets": delay.get("successful_packets", 0),
                "p50_ms": delay.get("p50_ms", ""), "p95_ms": delay.get("p95_ms", ""),
                "p99_ms": delay.get("p99_ms", ""), "max_ms": delay.get("max_ms", ""),
                "run_path": str(run.resolve()),
            })
    write_csv(args.output / "runs.csv", run_rows)
    write_csv(args.output / "ac_metrics.csv", ac_rows)
    print(json.dumps({"audited_runs": len(runs), "run_rows": len(run_rows), "ac_rows": len(ac_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
