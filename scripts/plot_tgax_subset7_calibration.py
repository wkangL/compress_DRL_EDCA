"""Create report figures for the formal 7-BSS TGax-based environment."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import RegularPolygon
import numpy as np

from build_tgax_s3_inputs import link_shadow_db, pathloss_db

AC_COLORS = {"VO": "#d62728", "VI": "#ff7f0e", "BE": "#1f77b4", "BK": "#9467bd"}
VARIANTS = ["mix_min1", "mix_min2", "load025", "load050", "load075", "load100", "load125", "load150", "rt125", "rt150"]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def save(fig, stem: str, output: Path) -> None:
    (output / "png").mkdir(parents=True, exist_ok=True)
    (output / "pdf").mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "png" / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "pdf" / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def topology_figure(input_root: Path, output: Path) -> None:
    topology = read(input_root / "topology.csv")
    aps = {int(r["bss"]): r for r in topology if r["role"] == "AP"}
    stas = defaultdict(list)
    for row in topology:
        if row["role"] == "STA":
            stas[int(row["bss"])].append(row)
    fig, ax = plt.subplots(figsize=(8.7, 7.2))
    for bss, ap in aps.items():
        x, y = float(ap["x"]), float(ap["y"])
        color = "#d62728" if bss == 0 else "#1f77b4"
        ax.add_patch(RegularPolygon((x, y), 6, radius=10, orientation=math.pi / 2,
                                    facecolor=color, edgecolor=color, alpha=.08))
        segments = [[(x, y), (float(s["x"]), float(s["y"]))] for s in stas[bss]]
        ax.add_collection(LineCollection(segments, colors="#9aa0a6", linewidths=.3, alpha=.3))
        ax.scatter([float(s["x"]) for s in stas[bss]], [float(s["y"]) for s in stas[bss]],
                   s=10, color="#4a4a4a", alpha=.7)
        ax.scatter(x, y, marker="^", s=150 if bss == 0 else 85, color=color,
                   edgecolor="white", linewidth=.8, zorder=4)
        ax.text(x, y + 2, f"BSS {bss}\n40 STA", ha="center", fontsize=8,
                weight="bold" if bss == 0 else "normal")
    ax.plot([-28, -8], [-27, -27], color="black", lw=2)
    ax.text(-18, -25.7, "20 m", ha="center", fontsize=8)
    ax.set(title="Formal TGax-based subset7 environment: actual ns-3 coordinates\n7 co-channel BSSs, 280 STAs; BSS 0 is the only future control target",
           xlabel="x / m", ylabel="y / m", aspect="equal")
    ax.text(.5, -.10, "Hexagons are TGax geometry reference cells, not measured or modeled AP coverage boundaries.",
            transform=ax.transAxes, ha="center", color="#8b0000", fontsize=8)
    ax.grid(alpha=.15)
    save(fig, "01_subset7_actual_topology", output)


def traffic_figure(input_root: Path, output: Path) -> None:
    flows = read(input_root / "load100" / "flows.csv")
    services = ["BV6", "BV3", "GMG", "VOIP", "FTP", "HTTP"]
    counts = defaultdict(lambda: defaultdict(int))
    ac_dir = defaultdict(int)
    target_sta = defaultdict(set)
    for row in flows:
        counts[int(row["bss"])][row["model"]] += 1
        ac_dir[(row["ac"], row["direction"])] += 1
        if row["bss"] == "0":
            endpoint = row["sink"] if row["direction"] == "DL" else row["source"]
            target_sta[endpoint].add(row["model"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    bsses = range(7); bottom = np.zeros(7)
    palette = dict(zip(services, plt.cm.tab10(np.linspace(0, .8, len(services)))))
    for service in services:
        values = np.asarray([counts[b][service] for b in bsses])
        axes[0].bar(list(bsses), values, bottom=bottom, label=service, color=palette[service])
        bottom += values
    axes[0].set(title="Official 100% application components", xlabel="BSS", ylabel="one-way components", xticks=list(bsses))
    axes[0].legend(frameon=False, ncol=2, fontsize=7)
    matrix = np.asarray([[ac_dir[(ac, direction)] for direction in ("DL", "UL")] for ac in ("VO", "VI", "BE")])
    im = axes[1].imshow(matrix, cmap="Blues")
    axes[1].set(title="All 7 BSSs: AC and direction", xticks=[0, 1], xticklabels=["DL", "UL"],
                yticks=[0, 1, 2], yticklabels=["VO", "VI", "BE"])
    for i in range(3):
        for j in range(2): axes[1].text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=axes[1], shrink=.75)
    sta_names = sorted(target_sta, key=lambda value: int(value.split("sta")[-1]))
    membership = np.asarray([[int(service in target_sta[name]) for service in services] for name in sta_names])
    axes[2].imshow(membership, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    axes[2].set(title="Target BSS: per-STA service assignment", xlabel="service", ylabel="STA index",
                xticks=range(len(services)), xticklabels=services, yticks=range(0, len(sta_names), 5),
                yticklabels=range(0, len(sta_names), 5))
    fig.suptitle("TGax Scenario 3 traffic models retained; real TCP/UDP is used")
    fig.tight_layout()
    save(fig, "02_subset7_business_assignment", output)


def interference_figure(input_root: Path, official_run: Path, output: Path, seed: int) -> None:
    active = read(input_root / "active_bss.csv")
    aps = [(int(r["bss"]), float(r["x"]), float(r["y"]), float(r["z"])) for r in active]
    matrix = np.full((7, 7), np.nan)
    for i, (_, x1, y1, z1) in enumerate(aps):
        for j, (_, x2, y2, z2) in enumerate(aps):
            if i != j:
                distance = math.dist((x1, y1, z1), (x2, y2, z2))
                matrix[i, j] = 23.0102999566 - pathloss_db(distance) - link_shadow_db((x1,y1,z1),(x2,y2,z2),seed)
    bss = {int(r["bss"]): r for r in read(official_run / "bss_summary.csv")}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5.2))
    im = ax0.imshow(matrix, cmap="viridis", vmin=-100, vmax=-45)
    ax0.set(title="AP-to-AP received power / dBm", xlabel="receiver BSS", ylabel="transmitter BSS",
            xticks=range(7), yticks=range(7))
    fig.colorbar(im, ax=ax0, shrink=.8)
    ids = range(7)
    busy = [float(bss[i]["busy_fraction"]) for i in ids]
    throughput = [float(bss[i]["throughput_mbps"]) for i in ids]
    ax1.bar(ids, busy, color=["#d62728"] + ["#1f77b4"] * 6, alpha=.75, label="AP non-idle fraction")
    ax1b = ax1.twinx(); ax1b.plot(ids, throughput, color="#2ca02c", marker="o", label="throughput")
    ax1.set(title="Measured competition under official 100% load", xlabel="BSS", ylabel="AP non-idle fraction", ylim=(0, 1))
    ax1b.set_ylabel("application throughput / Mbit/s")
    handles = ax1.get_legend_handles_labels()[0] + ax1b.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax1b.get_legend_handles_labels()[1]
    ax1.legend(handles, labels, frameon=False, fontsize=8)
    save(fig, "03_subset7_interference_and_competition", output)


def ac_load_figure(summary: Path, official_duration: float, output: Path) -> None:
    rows = [r for r in read(summary / "ac_metrics.csv")
            if r["variant"] == "load100" and r["scope"] == "TARGET" and float(r["measurement_s"]) == official_duration]
    keys = [(ac, direction) for ac in ("VO", "VI", "BE") for direction in ("DL", "UL")]
    lookup = {(r["ac"], r["direction"]): r for r in rows}
    labels = [f"{ac}-{direction}" for ac, direction in keys]
    offered = [float(lookup.get(k, {}).get("offered_bytes", 0)) * 8 / official_duration / 1e6 for k in keys]
    received = [float(lookup.get(k, {}).get("received_bytes", 0)) * 8 / official_duration / 1e6 for k in keys]
    x = np.arange(len(keys)); width = .36
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(x - width/2, offered, width, label="application generated", color="#9ecae1")
    ax.bar(x + width/2, received, width, label="application received", color="#2171b5")
    ax.set(title=f"Target BSS per-AC UL/DL load ({official_duration:g} s official 100% run)",
           xlabel="access category and direction", ylabel="rate / Mbit/s", xticks=x, xticklabels=labels)
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    save(fig, "04_target_ac_uplink_downlink_load", output)


def ccdf_figure(official_run: Path, output: Path) -> None:
    rows = [r for r in read(official_run / "mac_service_delay_ccdf.csv") if r["scope"] == "TARGET"]
    counts = {(r["direction"], r["ac"]): r["successful_packets"]
              for r in read(official_run / "mac_service_delay_summary.csv") if r["scope"] == "TARGET"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, direction in zip(axes, ("DL", "UL")):
        for ac in ("VO", "VI", "BE"):
            samples = [r for r in rows if r["direction"] == direction and r["ac"] == ac]
            if samples:
                ax.step([float(r["delay_ms"]) for r in samples], [float(r["ccdf"]) for r in samples],
                        where="post", color=AC_COLORS[ac], label=f"{ac} (n={counts[(direction, ac)]})")
        ax.set_yscale("log"); ax.set_ylim(5e-4, 1.05); ax.grid(True, which="both", alpha=.2)
        ax.set(title=direction, xlabel="final MAC service delay / ms")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("CCDF  P(delay >= x)")
    fig.suptitle("Target BSS Default EDCA: queue admission to successful ACK/Block ACK")
    fig.tight_layout()
    save(fig, "05_target_final_mac_service_delay_ccdf", output)


def load_scan_figure(summary: Path, output: Path) -> None:
    source = read(summary / "ac_metrics.csv")
    scan_duration = min(float(r["measurement_s"]) for r in source)
    rows = [r for r in source if r["scope"] == "TARGET"
            and float(r["measurement_s"]) == scan_duration]
    by_key = {(r["variant"], r["direction"], r["ac"]): r for r in rows}
    available = [v for v in VARIANTS if any((v, direction, ac) in by_key
                 for direction in ("DL", "UL") for ac in ("VO", "VI"))]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    def value(variant, direction, ac, field):
        raw = by_key.get((variant, direction, ac), {}).get(field, "")
        return float(raw) if raw not in ("", None) else np.nan
    for ac in ("VO", "VI"):
        color = AC_COLORS[ac]
        axes[0,0].plot(available, [value(v,"DL",ac,"p99_ms") for v in available], marker="o", color=color, label=ac)
        axes[0,1].plot(available, [value(v,"UL",ac,"p99_ms") for v in available], marker="o", color=color, label=ac)
        for direction, style in (("DL", "-"), ("UL", "--")):
            label = f"{ac}-{direction}"
            axes[1,0].plot(available, [value(v,direction,ac,"application_delivery_fraction") for v in available],
                           marker="o", ls=style, color=color, label=label)
            axes[1,1].plot(available, [value(v,direction,ac,"mac_failure_fraction") for v in available],
                           marker="o", ls=style, color=color, label=label)
    titles = ["DL p99 final MAC service delay", "UL p99 final MAC service delay",
              "application byte delivery fraction", "final MAC failure fraction"]
    ylabels = ["ms", "ms", "fraction", "fraction"]
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set(title=title, xlabel="load variant", ylabel=ylabel); ax.grid(alpha=.2); ax.tick_params(axis="x", rotation=25); ax.legend(frameon=False)
    fig.suptitle("Target BSS VO/VI response to TGax-based load-count extensions")
    fig.tight_layout()
    save(fig, "06_target_vo_vi_load_scan", output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--official-run", type=Path, required=True)
    parser.add_argument("--official-duration", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Microsoft YaHei","DejaVu Sans"],
                         "font.size":9, "axes.titlesize":11, "axes.labelsize":9, "pdf.fonttype":42})
    topology_figure(args.input, args.output)
    traffic_figure(args.input, args.output)
    interference_figure(args.input, args.official_run, args.output, args.seed)
    ac_load_figure(args.summary, args.official_duration, args.output)
    ccdf_figure(args.official_run, args.output)
    load_scan_figure(args.summary, args.output)


if __name__ == "__main__":
    main()
