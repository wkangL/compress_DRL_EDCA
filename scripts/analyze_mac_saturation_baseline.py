"""Analyze a direct-MAC, full-buffer TGax baseline run."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ACS = ("VO", "VI", "BE", "BK")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def div(a: float, b: float) -> float | None:
    return a / b if b else None


def jain(values: list[float]) -> float | None:
    square_sum = sum(value * value for value in values)
    return sum(values) ** 2 / (len(values) * square_sum) if values and square_sum else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--flows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--drain", type=float, default=1.0)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    flows = read(args.flows)
    outcomes = [r for r in read(run / "mac_qos_outcomes.csv") if r["scope"] == "BSS"]

    metric_rows = []
    for ac in ACS:
        for direction in ("DL", "UL", "BOTH"):
            selected = [r for r in outcomes if r["ac"] == ac and
                        (direction == "BOTH" or r["direction"] == direction)]
            arrived = sum(int(r["arrived"]) for r in selected)
            acked = sum(int(r["acked"]) for r in selected)
            failed = sum(int(r["failed"]) for r in selected)
            unfinished = sum(int(r["unfinished"]) for r in selected)
            attempts = sum(int(r["tx_attempts"]) for r in selected)
            retries = sum(int(r["retransmissions"]) for r in selected)
            acked_bytes = sum(int(r["acked_bytes"]) for r in selected)
            metric_rows.append({
                "ac": ac, "direction": direction, "arrived": arrived, "acked": acked,
                "failed": failed, "unfinished": unfinished, "acked_bytes": acked_bytes,
                "throughput_mbps": acked_bytes * 8 / args.duration / 1e6,
                "mac_failure_rate": div(failed, arrived),
                "unfinished_rate": div(unfinished, arrived),
                "tx_attempts": attempts, "retransmissions": retries,
                "retransmission_attempt_rate": div(retries, attempts),
            })
    write(output / "mac_metrics.csv", metric_rows)

    active = {(r["source"], r["ac"], r["direction"]) for r in flows}
    queue_samples = [r for r in read(run / "queue_samples.csv")
                     if args.warmup < float(r["time_s"]) <= args.warmup + args.duration]
    saturation_rows = []
    for node, ac, direction in sorted(active):
        selected = [r for r in queue_samples if r["node"] == node and r["ac"] == ac]
        fraction = div(sum(int(r["packets"]) > 0 for r in selected), len(selected))
        saturation_rows.append({
            "node": node, "ac": ac, "direction": direction, "samples": len(selected),
            "nonempty_fraction": fraction,
            "mean_packets": div(sum(int(r["packets"]) for r in selected), len(selected)),
            "min_packets": min((int(r["packets"]) for r in selected), default=None),
            "max_packets": max((int(r["packets"]) for r in selected), default=None),
            "full_buffer_95pct": fraction is not None and fraction >= .95,
        })
    write(output / "queue_saturation.csv", saturation_rows)

    ack_bytes = defaultdict(int)
    for row in read(run / "mac_packet_events.csv"):
        if (row["event"] == "ACKED" and row["ac"] in ACS and
                args.warmup <= float(row["time_s"]) <= args.warmup + args.duration + args.drain):
            ack_bytes[(row["node"], row["ac"])] += int(row["bytes"])
    fairness_rows = []
    for ac in ACS:
        for direction in ("DL", "UL"):
            nodes = sorted({node for node, item_ac, item_direction in active
                            if item_ac == ac and item_direction == direction})
            values = [ack_bytes[(node, ac)] * 8 / args.duration / 1e6 for node in nodes]
            fairness_rows.append({
                "ac": ac, "direction": direction, "active_source_queues": len(nodes),
                "zero_throughput_queues": sum(value == 0 for value in values),
                "jain_throughput": jain(values),
                "min_queue_throughput_mbps": min(values, default=None),
                "max_queue_throughput_mbps": max(values, default=None),
            })
    write(output / "source_queue_fairness.csv", fairness_rows)

    delay_rows = [r for r in read(run / "mac_service_delay_summary.csv") if r["scope"] == "BSS"]
    write(output / "delay_quantiles.csv", delay_rows)
    ccdf = [r for r in read(run / "mac_service_delay_ccdf.csv") if r["scope"] == "BSS"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax, ac in zip(axes.flat, ACS):
        plotted = False
        for direction, style in (("DL", "-"), ("UL", "--")):
            rows = [r for r in ccdf if r["ac"] == ac and r["direction"] == direction]
            count = next((int(r["successful_packets"]) for r in delay_rows
                          if r["ac"] == ac and r["direction"] == direction), 0)
            if rows:
                ax.plot([float(r["delay_ms"]) for r in rows],
                        [float(r["ccdf"]) for r in rows], style,
                        label=f"{direction} (n={count})")
                plotted = True
            elif count == 0:
                ax.text(.5, .18 if direction == "DL" else .08,
                        f"{direction}: no successful MPDU", transform=ax.transAxes,
                        ha="center", fontsize=9)
        ax.set_yscale("log")
        ax.set_title(ac)
        ax.set_xlabel("MAC service delay (ms)")
        ax.set_ylabel("P(delay > x)")
        ax.grid(True, which="both", alpha=.25)
        if plotted:
            ax.legend()
    fig.suptitle("Single-BSS direct-MAC full-buffer Default EDCA — delay CCDF")
    fig.savefig(output / "delay_ccdf_all_ac.png", dpi=180)
    plt.close(fig)

    associations = read(run / "associations.csv")
    checks = {
        "all_40_stas_associated": len(associations) == 40 and all(r["associated"] == "1" for r in associations),
        "all_bssid_matches": all(r["match"] == "1" for r in associations),
        "mac_ledger_conserved": all(int(r["arrived"]) == int(r["acked"]) + int(r["failed"]) + int(r["unfinished"])
                                    for r in outcomes),
        "all_four_acs_entered_mac": {r["ac"] for r in outcomes} == set(ACS),
        "all_active_queues_full_buffer_95pct": all(r["full_buffer_95pct"] for r in saturation_rows),
        "no_tcp_udp_ip_flow_results": not read(run / "ip_flow_summary.csv"),
    }
    validation = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    (output / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({
        "validation": validation["status"],
        "duration_s": args.duration,
        "total_throughput_mbps": sum(r["throughput_mbps"] for r in metric_rows if r["direction"] == "BOTH"),
        "busy_fraction": float(read(run / "bss_summary.csv")[0]["busy_fraction"]),
        "active_source_queues": len(saturation_rows),
        "minimum_queue_nonempty_fraction": min(r["nonempty_fraction"] for r in saturation_rows),
    }, indent=2), encoding="utf-8")
    if validation["status"] != "PASS":
        raise SystemExit("Validation failed")


if __name__ == "__main__":
    main()
