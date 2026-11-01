"""Analyze high-Gaming direct-MAC runs and compare the two arrival loads."""
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT
import scripts.analyze_mac_arrival40_default as analysis

analysis.RUN = ROOT / "results/raw/mac_arrival40_gaming160_default_20s_3seed_20261029"
analysis.INPUT = ROOT / "results/raw/tgax_mac_arrival40_gaming160_inputs_20261029"
analysis.OUT = ROOT / "results/summary/mac_arrival40_gaming160_default_20s_3seed_20261029"
analysis.FIG = ROOT / "results/figures/mac_arrival40_gaming160_default_20s_3seed_20261029"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def compare() -> None:
    base_dir = ROOT / "results/summary/mac_arrival40_default_20s_3seed_20261029"
    high = read(analysis.OUT / "per_seed_metrics.csv")
    base = read(base_dir / "per_seed_metrics.csv")
    metrics = (
        ("total_throughput_mbps", "总吞吐/Mbps"),
        ("busy_fraction", "信道忙比例"),
        ("UL_VO_p95_ms", "VO-UL p95/ms"),
        ("UL_VI_p95_ms", "VI-UL p95/ms"),
        ("UL_VO_on_time_rate", "VO及时率"),
        ("UL_VI_on_time_rate", "VI及时率"),
    )
    rows = []
    for key, label in metrics:
        before = np.mean([float(row[key]) for row in base])
        after = np.mean([float(row[key]) for row in high])
        rows.append({"metric": key, "label": label, "base_mean": before,
                     "gaming160_mean": after,
                     "change_percent": (after / before - 1) * 100 if before else None})
    with (analysis.OUT / "base_vs_gaming160.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), constrained_layout=True)
    labels = ["原随机到达", "Gaming×160"]
    axes[0].bar(labels, [rows[0]["base_mean"], rows[0]["gaming160_mean"]], color=["#4c72b0", "#c44e52"])
    axes[0].set(ylabel="Mbps", title="总成功MAC吞吐")
    axes[0].text(0, rows[0]["base_mean"], f'{rows[0]["base_mean"]:.2f}', ha="center", va="bottom")
    axes[0].text(1, rows[0]["gaming160_mean"], f'{rows[0]["gaming160_mean"]:.2f}', ha="center", va="bottom")
    x = np.arange(2); w = .34
    axes[1].bar(x-w/2, [rows[2]["base_mean"], rows[2]["gaming160_mean"]], w, label="VO-UL")
    axes[1].bar(x+w/2, [rows[3]["base_mean"], rows[3]["gaming160_mean"]], w, label="VI-UL")
    axes[1].set(xticks=x, xticklabels=labels, ylabel="ms", title="MAC最终服务时延p95")
    axes[1].legend()
    axes[2].bar(x-w/2, [rows[4]["base_mean"], rows[4]["gaming160_mean"]], w, label="VO≤20ms")
    axes[2].bar(x+w/2, [rows[5]["base_mean"], rows[5]["gaming160_mean"]], w, label="VI≤50ms")
    axes[2].set(xticks=x, xticklabels=labels, ylim=(0,1.03), ylabel="及时率", title="参考期限及时交付")
    axes[2].legend()
    for ax in axes: ax.grid(axis="y", alpha=.22)
    fig.savefig(analysis.FIG / "base_vs_gaming160.png", dpi=220); plt.close(fig)

    summary_path = analysis.OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["base_arrival_total_throughput_mbps_mean"] = rows[0]["base_mean"]
    summary["base_arrival_busy_fraction_mean"] = rows[1]["base_mean"]
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    analysis.main()
    compare()
