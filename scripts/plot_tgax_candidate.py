"""Plot the sole subset7 FTP-to-BK candidate from actual inputs/results."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, RegularPolygon
import numpy as np


COLORS = {"VO": "#d62728", "VI": "#ff7f0e", "BE": "#1f77b4", "BK": "#9467bd"}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def topology(input_root: Path, output: Path) -> None:
    rows = read(input_root / "topology.csv")
    aps = {int(r["bss"]): r for r in rows if r["role"] == "AP"}
    stas = defaultdict(list)
    for row in rows:
        if row["role"] == "STA":
            stas[int(row["bss"])].append(row)
    fig, ax = plt.subplots(figsize=(9.0, 7.7))
    for bss, ap in aps.items():
        x, y = float(ap["x"]), float(ap["y"])
        color = COLORS["VO"] if bss == 0 else COLORS["BE"]
        ax.add_patch(RegularPolygon((x, y), 6, radius=10, orientation=math.pi / 2,
                                    facecolor=color, edgecolor=color, alpha=.08, lw=1))
        segments = [[(x, y), (float(s["x"]), float(s["y"]))] for s in stas[bss]]
        ax.add_collection(LineCollection(segments, colors="#9aa0a6", linewidths=.25, alpha=.25))
        ax.scatter([float(s["x"]) for s in stas[bss]], [float(s["y"]) for s in stas[bss]],
                   s=9, color="#4a4a4a", alpha=.65)
        ax.scatter(x, y, marker="^", s=160 if bss == 0 else 90, color=color,
                   edgecolor="white", linewidth=.8, zorder=4)
        ax.text(x, y + 2, f"BSS {bss}\n40 STA", ha="center", fontsize=8,
                weight="bold" if bss == 0 else "normal")
    ax.plot([-28, -8], [-27, -27], color="black", lw=2)
    ax.text(-18, -25.7, "20 m", ha="center", fontsize=8)
    ax.legend(handles=[Patch(facecolor=COLORS["VO"], alpha=.25, label="目标BSS 0"),
                       Patch(facecolor=COLORS["BE"], alpha=.25, label="6个邻居BSS")],
              loc="upper right", frameon=False)
    ax.set(title="TGax-based subset7候选环境：ns-3实际AP/STA坐标与关联\n7个同信道BSS，中心AP 0为未来唯一控制对象",
           xlabel="x / m", ylabel="y / m", aspect="equal")
    ax.text(.5, -.10, "半透明六边形仅表示TGax几何参考小区，不是AP的实际无线覆盖边界。",
            transform=ax.transAxes, ha="center", color="#8b0000", fontsize=9)
    ax.grid(alpha=.15)
    save(fig, output / "01_candidate_actual_topology.png")


def assignment(summary: Path, output: Path) -> None:
    rows = read(summary / "bss0_sta_business_assignment.csv")
    active_count = sum(int(row["active_business_sta"]) for row in rows)
    services = ("D1", "D2", "D3", "D4", "D5", "D6-DL", "D6-UL", "U1", "U2", "U3")
    matrix = np.zeros((40, len(services)))
    ac_matrix = np.full((40, len(services)), np.nan)
    ac_index = {ac: i for i, ac in enumerate(("VO", "VI", "BE", "BK"))}
    for row in rows:
        sta = int(row["sta_index"])
        contents = f"{row['downlink_services']};{row['uplink_services']}"
        for col, service in enumerate(services):
            token = service.split("-")[0]
            direction_ok = (service == "D6-DL" and "D6(VOIP)" in row["downlink_services"] or
                            service == "D6-UL" and "D6(VOIP)" in row["uplink_services"] or
                            not service.startswith("D6-") and token + "(" in contents)
            if direction_ok:
                matrix[sta, col] = 1
                ac = row["application_acs"].split(";")[0].strip()
                ac_matrix[sta, col] = ac_index[ac]
    masked = np.ma.masked_invalid(ac_matrix)
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(masked, aspect="auto", interpolation="none",
              cmap=ListedColormap([COLORS[a] for a in ("VO", "VI", "BE", "BK")]), vmin=0, vmax=3)
    ax.set(title=f"BSS 0：40个关联STA中的实际应用业务分配（{active_count}个活跃STA）",
           xlabel="业务组件", ylabel="STA索引", xticks=range(len(services)), xticklabels=services,
           yticks=range(0, 40, 2))
    ax.grid(which="major", color="white", lw=.5, alpha=.4)
    ax.legend(handles=[Patch(color=COLORS[a], label=a) for a in ("VO", "VI", "BE", "BK")],
              ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.09))
    save(fig, output / "02_bss0_sta_business_assignment.png")


def metrics(summary: Path, output: Path, baseline_summary: Path | None) -> None:
    rows = read(summary / "bss0_ac_direction_metrics.csv")
    lookup = {(r["ac"], r["direction"]): r for r in rows}
    old_lookup = ({(r["ac"], r["direction"]): r
                   for r in read(baseline_summary / "bss0_ac_direction_metrics.csv")}
                  if baseline_summary else {})
    labels = [f"{ac}-{direction}" for ac in ("VO", "VI", "BE", "BK") for direction in ("DL", "UL")]
    keys = [(ac, direction) for ac in ("VO", "VI", "BE", "BK") for direction in ("DL", "UL")]
    colors = [COLORS[ac] for ac, _ in keys]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    fields = (("p95_final_mac_service_delay_ms", "MAC最终服务时延 p95", "ms"),
              ("successful_mac_throughput_mbps", "成功交付吞吐量", "Mbit/s"),
              ("final_mac_drop_rate", "MAC最终丢包率", "比例"),
              ("retransmission_rate", "重传率（碰撞代理量）", "重传/发射尝试"))
    x = np.arange(len(keys))
    for ax, (field, title, ylabel) in zip(axes.flat[:4], fields):
        values = [float(lookup[key][field] or 0) for key in keys]
        if old_lookup:
            old_values = [float(old_lookup[key][field] or 0) for key in keys]
            ax.bar(x - .2, old_values, .4, color="#bdbdbd", label="原candidate_ftp_bk")
            ax.bar(x + .2, values, .4, color=colors, alpha=.88, label="新VoIP8/Gaming4")
            ax.legend(frameon=False, fontsize=7)
        else:
            ax.bar(x, values, color=colors, alpha=.82)
        ax.set(title=title, ylabel=ylabel, xticks=x, xticklabels=labels)
        ax.tick_params(axis="x", rotation=30); ax.grid(axis="y", alpha=.2)
    validation = __import__("json").loads((summary / "candidate_validation.json").read_text(encoding="utf-8"))
    new_jain = float(validation["jain_demand_satisfaction"])
    new_active = int(validation["active_business_sta_count"])
    if baseline_summary:
        old_validation = __import__("json").loads(
            (baseline_summary / "candidate_validation.json").read_text(encoding="utf-8"))
        old_jain = float(old_validation["jain_demand_satisfaction"])
        old_active = int(old_validation["active_business_sta_count"])
        axes[1, 1].bar([0, 1], [old_jain, new_jain], color=["#bdbdbd", "#2ca02c"])
        axes[1, 1].set(xticks=[0, 1], xticklabels=["原候选", "新并发"], ylim=(0, 1.05))
        axes[1, 2].bar([0, 1], [old_active, new_active], color=["#bdbdbd", "#17becf"])
        axes[1, 2].set(xticks=[0, 1], xticklabels=["原候选", "新并发"], ylim=(0, 40))
    else:
        axes[1, 1].bar([0], [new_jain], color="#2ca02c")
        axes[1, 1].set(xticks=[0], xticklabels=["新并发"], ylim=(0, 1.05))
        axes[1, 2].bar([0], [new_active], color="#17becf")
        axes[1, 2].set(xticks=[0], xticklabels=["新并发"], ylim=(0, 40))
    axes[1, 1].set(title="Jain需求满足公平性", ylabel="Jain指数"); axes[1, 1].grid(axis="y", alpha=.2)
    axes[1, 2].set(title="实际承载应用业务STA", ylabel="STA数量"); axes[1, 2].grid(axis="y", alpha=.2)
    fig.suptitle("BSS 0 Default EDCA：20秒Baseline五类核心指标与活跃STA")
    fig.tight_layout()
    save(fig, output / "03_bss0_default_edca_metrics.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
                         "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
    topology(args.input_root, args.output)
    assignment(args.summary, args.output)
    metrics(args.summary, args.output, args.baseline_summary)


if __name__ == "__main__":
    main()
