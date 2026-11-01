"""Audit and visualize the sole 40-STA multi-AC TGax research scenario."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT


ACS = ("VO", "VI", "BE", "BK")
COLORS = {"VO": "#d62728", "VI": "#ff9f1c", "BE": "#2878b5", "BK": "#7b4ab0"}
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def expand_ccdf_samples(rows: list[dict[str, str]], successful_packets: int) -> list[float]:
    """Reconstruct weighted samples from the deliberately thinned ns-3 CCDF.

    tgax-s3 keeps at most about 2000 sorted points per direction/AC and writes
    each retained point's exact empirical CCDF.  Treating retained rows as an
    unweighted sample biases a UL+DL percentile whenever their thinning strides
    differ.  This function restores each point's represented rank interval.
    The resulting percentile is exact when stride=1 and bounded by one thinning
    interval otherwise.  Online STEP evaluation uses every Y event and is exact.
    """
    if not rows or successful_packets <= 0:
        return []
    points = []
    for row in rows:
        rank = int(round(successful_packets * (1.0 - float(row["ccdf"]))))
        points.append((max(0, min(successful_packets - 1, rank)),
                       float(row["delay_ms"])))
    points.sort(key=lambda item: item[0])
    expanded: list[float] = []
    for index, (rank, value) in enumerate(points):
        next_rank = points[index + 1][0] if index + 1 < len(points) else successful_packets
        expanded.extend([value] * max(0, next_rank - rank))
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flows", type=Path, default=ROOT / (
        "results/raw/tgax_s3_inputs_20261021/subset7/candidate_full40_multiac_v2/flows.csv"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    flows = read(args.flows)
    bss0 = [row for row in flows if row["bss"] == "0"]

    cells: dict[tuple[int, str], list[str]] = defaultdict(list)
    for row in bss0:
        label = row["service"]
        if label not in cells[(int(row["sta"]), row["ac"])]:
            cells[(int(row["sta"]), row["ac"])].append(label)
    matrix = np.zeros((40, 4), dtype=float)
    for sta in range(40):
        for ac_index, ac in enumerate(ACS):
            matrix[sta, ac_index] = 1 if cells[(sta, ac)] else np.nan
    fig, ax = plt.subplots(figsize=(8.2, 12.0))
    cmap = ListedColormap(["#ffffff", "#555555"])
    rgba = np.ones((40, 4, 4))
    for sta in range(40):
        for ac_index, ac in enumerate(ACS):
            if cells[(sta, ac)]:
                rgba[sta, ac_index] = (*plt.matplotlib.colors.to_rgb(COLORS[ac]), 0.88)
            else:
                rgba[sta, ac_index] = (0.96, 0.96, 0.96, 1.0)
    ax.imshow(rgba, aspect="auto", interpolation="none")
    for sta in range(40):
        for ac_index, ac in enumerate(ACS):
            labels = cells[(sta, ac)]
            ax.text(ac_index, sta, "\n".join(labels), ha="center", va="center",
                    fontsize=6.4, color="white" if labels else "#bbbbbb")
    ax.set(xticks=range(4), xticklabels=ACS, yticks=range(40),
           yticklabels=[f"STA {i:02d}" for i in range(40)],
           xlabel="接入类别（AC）", ylabel="BSS 0固定STA编号",
           title="candidate_full40_multiac_v2：40 STA × 4 AC固定业务覆盖\n"
                 "每个STA均有上行VO或VI；彩色单元格中的文字为实际业务服务标识")
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 40, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(args.output / "bss0_40sta_4ac_coverage.png", dpi=240)
    plt.close(fig)

    outcomes = [row for row in read(args.run / "mac_qos_outcomes.csv")
                if row["scope"] == "BSS" and row["bss"] == "0"]
    delay_rows = [row for row in read(args.run / "mac_service_delay_ccdf.csv")
                  if row["scope"] == "BSS" and row["bss"] == "0"]
    queues = [row for row in read(args.run / "queue_summary.csv") if row["bss"] == "0"]
    summary = []
    for ac in ACS:
        group = [row for row in outcomes if row["ac"] == ac]
        delays = []
        retained_ccdf_points = 0
        for direction in ("UL", "DL"):
            outcome = next((row for row in group if row["direction"] == direction), None)
            rows = [row for row in delay_rows
                    if row["ac"] == ac and row["direction"] == direction]
            retained_ccdf_points += len(rows)
            delays.extend(expand_ccdf_samples(rows, int(outcome["acked"]) if outcome else 0))
        arrived = sum(int(row["arrived"]) for row in group)
        acked = sum(int(row["acked"]) for row in group)
        failed = sum(int(row["failed"]) for row in group)
        unfinished = sum(int(row["unfinished"]) for row in group)
        attempts = sum(int(row["tx_attempts"]) for row in group)
        retries = sum(int(row["retransmissions"]) for row in group)
        acked_bytes = sum(int(row["acked_bytes"]) for row in group)
        queue_group = [row for row in queues if row["ac"] == ac]
        summary.append({
            "ac": ac, "arrived_mpdus": arrived, "acked_mpdus": acked,
            "failed_mpdus": failed, "unfinished_mpdus": unfinished,
            "successful_samples": acked,
            "retained_ccdf_points": retained_ccdf_points,
            "combined_delay_p95_ms": float(np.percentile(delays, 95)) if delays else None,
            "combined_delay_p95_basis": ("exact_full_samples" if retained_ccdf_points == acked
                                          else "rank_weighted_thinned_ccdf"),
            "successful_throughput_mbps": acked_bytes * 8 / 1e6 / args.duration,
            "final_failure_rate": failed / arrived if arrived else None,
            "retransmission_rate": retries / attempts if attempts else None,
            "mean_queue_packets_unweighted_direction": float(np.mean(
                [float(row["mean_packets"]) for row in queue_group])) if queue_group else None,
            "max_queue_packets": max((int(row["max_packets"]) for row in queue_group), default=0),
        })
    write(args.output / "default_bss0_ac_summary.csv", summary)

    coverage = []
    for bss in range(7):
        group = [row for row in flows if int(row["bss"]) == bss]
        high = {int(row["sta"]) for row in group
                if row["direction"] == "UL" and row["ac"] in {"VO", "VI"}}
        coverage.append({"bss": bss, "uplink_vo_or_vi_sta": len(high),
                         "be_sta": len({int(row["sta"]) for row in group if row["ac"] == "BE"}),
                         "bk_sta": len({int(row["sta"]) for row in group if row["ac"] == "BK"}),
                         "one_way_components": len(group)})
    write(args.output / "coverage_audit.csv", coverage)
    checks = {
        "all_bss_have_40_uplink_vo_or_vi_stas": all(row["uplink_vo_or_vi_sta"] == 40 for row in coverage),
        "all_bss_have_32_be_stas": all(row["be_sta"] == 32 for row in coverage),
        "all_bss_have_16_bk_stas": all(row["bk_sta"] == 16 for row in coverage),
        "all_four_acs_arrive_at_bss0_mac": {row["ac"] for row in summary if row["arrived_mpdus"] > 0} == set(ACS),
        "all_four_acs_successfully_delivered_at_bss0": {row["ac"] for row in summary if row["acked_mpdus"] > 0} == set(ACS),
        "mac_ledger_conserved": all(row["arrived_mpdus"] == row["acked_mpdus"] +
                                    row["failed_mpdus"] + row["unfinished_mpdus"] for row in summary),
    }
    (args.output / "scenario_audit.json").write_text(
        json.dumps({"status": "PASS" if all(checks.values()) else "FAIL",
                    "checks": checks, "duration_s": args.duration,
                    "flow_components_network": len(flows)}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == "__main__":
    main()
