"""Summarize one audited TGax Default-EDCA run into reproducible baseline metrics."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt


ACS = ("VO", "VI", "BE", "BK")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def jain(values: list[float]) -> float | None:
    if not values or sum(value * value for value in values) == 0:
        return None
    return sum(values) ** 2 / (len(values) * sum(value * value for value in values))


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    validation = json.loads((run / "calibration_validation.json").read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise RuntimeError("Run has not passed calibration validation")
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    load_manifest = json.loads((run / "load_manifest.json").read_text(encoding="utf-8"))
    duration = float(manifest["measurement_s"])
    flows = read_csv(run / "flow_summary.csv")
    qos = read_csv(run / "mac_qos_outcomes.csv")
    delays = read_csv(run / "mac_service_delay_summary.csv")
    ccdf = read_csv(run / "mac_service_delay_ccdf.csv")

    throughput_rows: list[dict] = []
    for scope, bss in (("ALL", None), ("BSS0", 0)):
        scoped = flows if bss is None else [r for r in flows if int(r["bss"]) == bss]
        for ac in (*ACS, "ALL"):
            for direction in ("DL", "UL", "BOTH"):
                selected = [r for r in scoped
                            if (ac == "ALL" or r["ac"] == ac)
                            and (direction == "BOTH" or r["direction"] == direction)]
                offered = sum(int(r["offered_bytes"]) for r in selected)
                received = sum(int(r["received_bytes"]) for r in selected)
                throughput_rows.append({
                    "scope": scope, "ac": ac, "direction": direction,
                    "offered_bytes": offered, "received_bytes": received,
                    "throughput_mbps": received * 8 / duration / 1e6,
                    "application_delivery_fraction": ratio(received, offered),
                    "application_delivery_deficit": 1 - received / offered if offered else None,
                })
    write_csv(output / "throughput_and_delivery.csv", throughput_rows)

    fairness_rows: list[dict] = []
    for scope, bss in (("ALL", None), ("BSS0", 0)):
        scoped = flows if bss is None else [r for r in flows if int(r["bss"]) == bss]
        for ac in (*ACS, "ALL"):
            selected = [r for r in scoped if ac == "ALL" or r["ac"] == ac]
            flow_tp = [int(r["received_bytes"]) * 8 / duration / 1e6 for r in selected]
            flow_sat = [int(r["received_bytes"]) / int(r["offered_bytes"])
                        for r in selected if int(r["offered_bytes"])]
            sta = defaultdict(lambda: [0, 0])
            for r in selected:
                key = (int(r["bss"]), r["sink"] if r["direction"] == "DL" else r["source"])
                sta[key][0] += int(r["offered_bytes"])
                sta[key][1] += int(r["received_bytes"])
            sta_tp = [received * 8 / duration / 1e6 for offered, received in sta.values()]
            sta_sat = [received / offered for offered, received in sta.values() if offered]
            fairness_rows.append({
                "scope": scope, "ac": ac, "flow_count": len(selected), "sta_count": len(sta),
                "flow_throughput_jain": jain(flow_tp), "flow_satisfaction_jain": jain(flow_sat),
                "sta_throughput_jain": jain(sta_tp), "sta_satisfaction_jain": jain(sta_sat),
            })
    write_csv(output / "fairness.csv", fairness_rows)

    saturation_rows: list[dict] = []
    queue_samples_path = run / "queue_samples.csv"
    if queue_samples_path.is_file() and queue_samples_path.stat().st_size > 100:
        samples = read_csv(queue_samples_path)
        warmup = float(manifest["warmup_s"])
        measure_end = warmup + duration
        samples = [r for r in samples if warmup < float(r["time_s"]) <= measure_end]
        for ac in ACS:
            for direction in ("DL", "UL"):
                active_flows = [r for r in flows if r["ac"] == ac and r["direction"] == direction]
                nodes = sorted({r["source"] for r in active_flows})
                fractions = []
                for node in nodes:
                    selected = [r for r in samples if r["node"] == node and r["ac"] == ac]
                    if selected:
                        fractions.append(sum(int(r["packets"]) > 0 for r in selected) / len(selected))
                saturation_rows.append({
                    "ac": ac, "direction": direction, "active_edca_queues": len(fractions),
                    "queue_samples": sum(1 for node in nodes for r in samples
                                         if r["node"] == node and r["ac"] == ac),
                    "mean_nonempty_fraction": sum(fractions) / len(fractions) if fractions else None,
                    "median_nonempty_fraction": median(fractions) if fractions else None,
                    "min_nonempty_fraction": min(fractions) if fractions else None,
                    "max_nonempty_fraction": max(fractions) if fractions else None,
                    "all_queues_full_buffer_95pct": bool(fractions) and min(fractions) >= .95,
                })
        write_csv(output / "queue_saturation.csv", saturation_rows)

    mac_rows: list[dict] = []
    for scope_name, source_scope in (("ALL", "ALL"), ("BSS0", "BSS")):
        selected_scope = [r for r in qos if r["scope"] == source_scope and
                          (scope_name == "ALL" or int(r["bss"]) == 0)]
        for ac in ACS:
            for direction in ("DL", "UL", "BOTH"):
                selected = [r for r in selected_scope if r["ac"] == ac and
                            (direction == "BOTH" or r["direction"] == direction)]
                arrived = sum(int(r["arrived"]) for r in selected)
                acked = sum(int(r["acked"]) for r in selected)
                failed = sum(int(r["failed"]) for r in selected)
                unfinished = sum(int(r["unfinished"]) for r in selected)
                attempts = sum(int(r["tx_attempts"]) for r in selected)
                retries = sum(int(r["retransmissions"]) for r in selected)
                mac_rows.append({
                    "scope": scope_name, "ac": ac, "direction": direction,
                    "arrived": arrived, "acked": acked, "failed": failed,
                    "unfinished": unfinished, "tx_attempts": attempts,
                    "retransmissions": retries, "mac_final_failure_rate": ratio(failed, arrived),
                    "mac_unfinished_rate": ratio(unfinished, arrived),
                    "retransmission_attempt_rate": ratio(retries, attempts),
                    "retries_per_arrived_mpdu": ratio(retries, arrived),
                })
    write_csv(output / "mac_loss_and_retransmission.csv", mac_rows)

    delay_rows = [r for r in delays if r["scope"] == "ALL" or
                  (r["scope"] == "BSS" and r["bss"] == "0")]
    for row in delay_rows:
        row["scope"] = "BSS0" if row["scope"] == "BSS" else "ALL"
    write_csv(output / "delay_quantiles.csv", delay_rows)

    active_acs = [ac for ac in ACS if any(r["ac"] == ac for r in flows)]
    if len(active_acs) == 4:
        fig, axes_grid = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
        axes = list(axes_grid.flat)
    else:
        fig, axes_grid = plt.subplots(1, len(active_acs),
                                      figsize=(5.5 * len(active_acs), 4.3),
                                      constrained_layout=True)
        axes = [axes_grid] if len(active_acs) == 1 else list(axes_grid)
    for ax, ac in zip(axes, active_acs):
        plotted = False
        for direction, style in (("DL", "-"), ("UL", "--")):
            rows = [r for r in ccdf if r["scope"] == "ALL" and r["ac"] == ac
                    and r["direction"] == direction]
            if rows:
                packet_count = next(int(r["successful_packets"]) for r in delays
                                    if r["scope"] == "ALL" and r["ac"] == ac
                                    and r["direction"] == direction)
                ax.plot([float(r["delay_ms"]) for r in rows],
                        [float(r["ccdf"]) for r in rows], style,
                        label=f"{direction} (n={packet_count})")
                plotted = True
        ax.set_yscale("log")
        ax.set_title(ac)
        ax.set_xlabel("MAC service delay (ms)")
        ax.set_ylabel("P(delay > x)")
        ax.grid(True, which="both", alpha=.25)
        if plotted:
            ax.legend()
    active_bss = int(load_manifest.get("active_bss", len({r["bss"] for r in flows})))
    fig.suptitle(f"Default EDCA delay CCDF — {active_bss} BSS, successful MPDUs only")
    fig.savefig(output / "delay_ccdf_all_ac.png", dpi=180)
    plt.close(fig)

    def pick(rows: list[dict], scope: str, ac: str, direction: str = "BOTH") -> dict:
        return next(r for r in rows if r["scope"] == scope and r["ac"] == ac
                    and r.get("direction", direction) == direction)

    summary = {
        "validation": validation["status"], "measurement_s": duration,
        "seed": manifest["seed"], "wall_seconds": manifest["elapsed_wall_seconds"],
        "network_throughput_mbps": pick(throughput_rows, "ALL", "ALL")["throughput_mbps"],
        "bss0_throughput_mbps": pick(throughput_rows, "BSS0", "ALL")["throughput_mbps"],
        "network_application_delivery_deficit": pick(throughput_rows, "ALL", "ALL")["application_delivery_deficit"],
        "bss0_application_delivery_deficit": pick(throughput_rows, "BSS0", "ALL")["application_delivery_deficit"],
        "network_sta_throughput_jain": pick(fairness_rows, "ALL", "ALL")["sta_throughput_jain"],
        "network_sta_satisfaction_jain": pick(fairness_rows, "ALL", "ALL")["sta_satisfaction_jain"],
        "bss0_sta_throughput_jain": pick(fairness_rows, "BSS0", "ALL")["sta_throughput_jain"],
        "bss0_sta_satisfaction_jain": pick(fairness_rows, "BSS0", "ALL")["sta_satisfaction_jain"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
