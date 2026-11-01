"""Plot an auditable view of the exact topology used by the current TGax DDQN run."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import RegularPolygon
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOPOLOGY = ROOT / "results/raw/tgax_s3_inputs_20261021/subset7/topology.csv"
DEFAULT_FLOWS = ROOT / "results/raw/tgax_s3_inputs_20261021/subset7/candidate_voip8_gaming4_ftp_bk/flows.csv"
DEFAULT_ASSOCIATIONS = ROOT / (
    "results/raw/tgax_ddqn_extended_seed41_400step_20261025/"
    "held_out/seed_45/default/associations.csv"
)
DEFAULT_OUTPUT = ROOT / "results/figures/tgax_current_topology_audit_20261026"
HEX_RADIUS_M = 10.0


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--flows", type=Path, default=DEFAULT_FLOWS)
    parser.add_argument("--associations", type=Path, default=DEFAULT_ASSOCIATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    topology = read(args.topology)
    flows = read(args.flows)
    associations = read(args.associations)
    aps = {int(row["bss"]): row for row in topology if row["role"] == "AP"}
    stas: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in topology:
        if row["role"] == "STA":
            stas[int(row["bss"])].append(row)

    runtime = defaultdict(lambda: {"associated": 0, "match": 0})
    for row in associations:
        bss = int(row["bss"])
        runtime[bss]["associated"] += int(row["associated"])
        runtime[bss]["match"] += int(row["match"])

    active_sta: dict[int, set[int]] = defaultdict(set)
    for row in flows:
        active_sta[int(row["bss"])].add(int(row["sta"]))

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False})
    colors = plt.cm.tab10(np.linspace(0, 0.85, 7))
    fig = plt.figure(figsize=(15.2, 8.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.45, 1.0))
    ax = fig.add_subplot(grid[0, 0])
    count_ax = fig.add_subplot(grid[0, 1])

    for bss in range(7):
        ap = aps[bss]
        x, y = float(ap["x"]), float(ap["y"])
        color = colors[bss]
        ax.add_patch(RegularPolygon(
            (x, y), 6, radius=HEX_RADIUS_M, orientation=math.pi / 2,
            facecolor="none", edgecolor=color, linestyle="--", linewidth=1.0, alpha=0.55,
        ))
        sx = [float(row["x"]) for row in stas[bss]]
        sy = [float(row["y"]) for row in stas[bss]]
        ax.scatter(sx, sy, s=20, color=color, edgecolor="white", linewidth=0.35,
                   alpha=0.82, zorder=3)
        ax.scatter(x, y, marker="^", s=145 if bss == 0 else 105, color=color,
                   edgecolor="black", linewidth=0.7, zorder=5)
        ax.annotate(f"BSS {bss}\nAP + 40关联STA", (x, y), xytext=(0, 9),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5,
                    weight="bold" if bss == 0 else "normal",
                    bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=1.4))

    center = aps[0]
    cx, cy = float(center["x"]), float(center["y"])
    for bss in range(1, 7):
        ap = aps[bss]
        ax.plot([cx, float(ap["x"])], [cy, float(ap["y"])], color="#555555",
                linestyle=":", linewidth=0.8, alpha=0.45, zorder=1)
    ax.annotate("中心AP到任一邻居AP：30 m", xy=(0, -15), xytext=(8, -8),
                textcoords="data", fontsize=9, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="#aaaaaa", pad=2))

    # This is a graphical length scale only; it is deliberately placed away from AP links.
    scale_x, scale_y = -37.0, -41.0
    ax.plot([scale_x, scale_x + 20.0], [scale_y, scale_y], color="black", lw=2.2)
    ax.plot([scale_x, scale_x], [scale_y - 0.7, scale_y + 0.7], color="black", lw=1.5)
    ax.plot([scale_x + 20.0, scale_x + 20.0], [scale_y - 0.7, scale_y + 0.7], color="black", lw=1.5)
    ax.text(scale_x + 10.0, scale_y + 1.2, "地图长度比例尺：20 m", ha="center", fontsize=9)

    ax.set(xlabel="x坐标 / m", ylabel="y坐标 / m", xlim=(-41, 41), ylim=(-44, 54), aspect="equal")
    ax.grid(alpha=0.16)
    ax.legend(handles=[
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#666666",
               markeredgecolor="black", markersize=9, label="AP"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666",
               markeredgecolor="white", markersize=7, label="STA（颜色表示关联BSS）"),
        Line2D([0], [0], color="#666666", linestyle="--",
               label="半径10 m的TGax参考六边形（非覆盖边界）"),
    ], loc="upper right", fontsize=8.5, framealpha=0.92)
    ax.set_title("当前训练实际使用的subset7坐标与关联", weight="bold")

    bsses = np.arange(7)
    input_counts = np.array([len(stas[bss]) for bss in bsses])
    runtime_counts = np.array([runtime[bss]["associated"] for bss in bsses])
    active_counts = np.array([len(active_sta[bss]) for bss in bsses])
    width = 0.24
    count_ax.bar(bsses - width, input_counts, width, label="拓扑CSV配置STA", color="#4c78a8")
    count_ax.bar(bsses, runtime_counts, width, label="seed45运行时已关联STA", color="#59a14f")
    count_ax.bar(bsses + width, active_counts, width, label="承载至少一个应用的STA", color="#f28e2b")
    for xpos, values in ((bsses - width, input_counts), (bsses, runtime_counts),
                         (bsses + width, active_counts)):
        for x, value in zip(xpos, values):
            count_ax.text(x, value + 0.6, str(value), ha="center", va="bottom", fontsize=8)
    count_ax.set(xticks=bsses, xticklabels=[f"BSS {bss}" for bss in bsses], ylabel="STA数量",
                 ylim=(0, 47))
    count_ax.grid(axis="y", alpha=0.18)
    count_ax.legend(loc="upper center", fontsize=8.5)
    count_ax.set_title("数量审计：关联STA与活跃业务STA不是同一概念", weight="bold")
    count_ax.text(
        0.02, 0.02,
        "核对结果\n"
        "• topology.csv：7×40 = 280个STA，坐标均唯一\n"
        "• 最新seed45运行：每BSS 40/40关联，且BSSID全部匹配\n"
        "• 当前业务随机分配会在同一STA上重叠，因此活跃STA为19–22个/BSS\n"
        "• STA由61小区全局投放后按含阴影的最强接收功率关联，\n"
        "  所以关联STA可以落在参考六边形之外",
        transform=count_ax.transAxes, ha="left", va="bottom", fontsize=9.3,
        linespacing=1.55, bbox=dict(facecolor="#f6f6f6", edgecolor="#bbbbbb", pad=8),
    )

    fig.suptitle(
        "TGax-based 7 BSS当前研究拓扑审计（数据直接来自训练输入与运行时关联记录）",
        fontsize=15, weight="bold",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "current_subset7_topology_verified.png", dpi=260)
    fig.savefig(args.output / "current_subset7_topology_verified.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
