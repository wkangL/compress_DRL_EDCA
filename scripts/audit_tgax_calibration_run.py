"""Audit a summary-mode TGax calibration run without relying on debug traces."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--flows", type=Path, required=True)
    parser.add_argument("--expected-stas", type=int, required=True)
    parser.add_argument("--scene", required=True)
    args = parser.parse_args()
    run = args.run
    associations = read(run / "associations.csv")
    outcomes = [row for row in read(run / "mac_qos_outcomes.csv") if row["scope"] == "BSS"]
    delays = read(run / "mac_service_delay_summary.csv")
    flows = read(args.flows)
    flow_summary = read(run / "flow_summary.csv")
    http_count = sum(row["model"] == "HTTP" for row in flows)

    checks = {
        f"all_{args.expected_stas}_stas_reported": len(associations) == args.expected_stas,
        "all_stas_associated": all(row["associated"] == "1" for row in associations),
        "all_bssid_matches": all(row["match"] == "1" for row in associations),
        "all_application_flows_reported": len(flow_summary) == len(flows) + http_count,
        "mac_outcome_groups_present": bool(outcomes),
        "mac_packet_conservation": all(
            int(row["arrived"]) == int(row["acked"]) + int(row["failed"]) + int(row["unfinished"])
            for row in outcomes
        ),
        "outcome_fractions_bounded": all(
            0 <= float(row[field]) <= 1
            for row in outcomes
            for field in ("ack_fraction", "failure_fraction", "unfinished_fraction")
        ),
        "retransmission_accounting_valid": all(
            int(row["retransmissions"]) <= int(row["tx_attempts"])
            for row in outcomes
        ),
        "delay_quantiles_monotonic": all(
            float(row["p50_ms"]) <= float(row["p95_ms"]) <= float(row["p99_ms"]) <= float(row["max_ms"])
            for row in delays if int(row["successful_packets"]) > 0
        ),
        "queue_summary_present": (run / "queue_summary.csv").stat().st_size > 80,
        "runtime_edca_exported": (run / "edca_runtime.csv").stat().st_size > 100,
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    (run / "calibration_validation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [f"# {args.scene} calibration validation: {result['status']}", ""]
    lines += [f"- [{'x' if passed else ' '}] {name}" for name, passed in checks.items()]
    (run / "calibration_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
