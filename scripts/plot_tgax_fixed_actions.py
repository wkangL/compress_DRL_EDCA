#!/usr/bin/env python3
"""Combine paired fixed-action runs and create the small set of report figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NAMES = {
    "A_default_beb": "A: P0 + native BEB",
    "B_default_custom": "B: P0 + custom CW",
    "C_p8_beb": "C: P8 + native BEB",
    "D_p8_custom": "D: P8 + custom CW",
}
COLORS = {
    "A_default_beb": "#3b6fb6",
    "B_default_custom": "#df7f2a",
    "C_p8_beb": "#3a9d5d",
    "D_p8_custom": "#b54b5f",
}
PAIRS = [("B_default_custom", "A_default_beb", "B-A"),
         ("C_p8_beb", "A_default_beb", "C-A"),
         ("D_p8_custom", "C_p8_beb", "D-C")]
ACS = ["vo", "vi", "be", "bk"]
DIRS = ["ul", "dl"]


def finite(x: object) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def load_runs(run_dirs: list[Path]) -> tuple[list[dict], list[dict]]:
    rows, summaries = [], []
    for root in run_dirs:
        with (root / "comparison.csv").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                converted = dict(row)
                for key, value in row.items():
                    if key in {"name", "description", "action", "custom_growth"} or value == "":
                        continue
                    try:
                        converted[key] = float(value)
                    except ValueError:
                        pass
                rows.append(converted)
        for name in NAMES:
            p = root / name / "fixed_action_summary.json"
            with p.open(encoding="utf-8") as f:
                summaries.append(json.load(f))
    return rows, summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tables(rows_in: list[dict], summaries: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "combined_metrics.csv", rows_in)
    metric_cols = [c for c in rows_in[0] if c not in {"name", "description", "action", "custom_growth", "seed"}]
    mean_rows = []
    for name in NAMES:
        group = [r for r in rows_in if r["name"] == name]
        row = {"name": name}
        for metric in metric_cols:
            vals = [float(r[metric]) for r in group if finite(r.get(metric))]
            row[metric] = float(np.mean(vals)) if vals else ""
        mean_rows.append(row)
    write_csv(out / "two_seed_means.csv", mean_rows)

    rows = []
    for seed in sorted({int(float(r["seed"])) for r in rows_in}):
        by_name = {r["name"]: r for r in rows_in if int(float(r["seed"])) == seed}
        for lhs, rhs, effect in PAIRS:
            for metric in metric_cols:
                a, b = by_name[lhs].get(metric), by_name[rhs].get(metric)
                if finite(a) and finite(b) and float(b) != 0:
                    pct = (float(a) - float(b)) / abs(float(b)) * 100.0
                else:
                    pct = math.nan
                rows.append({"seed": seed, "effect": effect, "metric": metric,
                             "candidate": a, "reference": b, "change_percent": pct})
    write_csv(out / "paired_effects.csv", rows)

    app_rows = []
    for s in summaries:
        for flow, vals in s["application_delivery"].items():
            offered, received = vals["offered_bytes"], vals["received_bytes"]
            app_rows.append({"seed": s["seed"], "name": s["name"], "flow": flow,
                             "offered_bytes": offered, "received_bytes": received,
                             "delivery_fraction": received / offered if offered else math.nan})
    write_csv(out / "application_delivery.csv", app_rows)


def read_delays(root: Path, name: str) -> dict[tuple[str, str], list[float]]:
    result = {(d, a): [] for d in DIRS for a in ["vo", "vi"]}
    dmap, amap = {0: "ul", 1: "dl"}, {0: "vo", 1: "vi", 2: "be", 3: "bk"}
    with (root / name / "events.log").open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("Y "):
                continue
            p = line.split()
            if len(p) < 6:
                continue
            t, direction, ac, delay = float(p[1]), int(p[2]), int(p[3]), float(p[4])
            # 1 s common traffic warmup, 20 s measurement; drain is excluded.
            if not (1.0 < t <= 21.0):
                continue
            key = (dmap.get(direction, ""), amap.get(ac, ""))
            if key in result:
                result[key].append(delay)
    return result


def plot_ccdf(run_dirs: list[Path], fig_dir: Path) -> None:
    pooled = {name: {(d, a): [] for d in DIRS for a in ["vo", "vi"]} for name in NAMES}
    for root in run_dirs:
        for name in NAMES:
            current = read_delays(root, name)
            for key, values in current.items():
                pooled[name][key].extend(values)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), constrained_layout=True)
    for ax, key in zip(axes.flat, [("ul", "vo"), ("dl", "vo"), ("ul", "vi"), ("dl", "vi")]):
        for name in NAMES:
            values = np.sort(np.asarray(pooled[name][key], dtype=float))
            if values.size == 0:
                continue
            ccdf = (values.size - np.arange(values.size)) / values.size
            ax.step(values, ccdf, where="post", lw=1.5, color=COLORS[name], label=NAMES[name])
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.05)
        ax.set_xlabel("MAC final service delay (ms)")
        ax.set_ylabel("CCDF  P(delay > x)")
        ax.set_title(f"{key[0].upper()} {key[1].upper()} (pooled seeds 41, 42)")
        ax.grid(True, which="both", alpha=.25)
    axes[0, 0].legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"vo_vi_delay_ccdf.{ext}", dpi=220)
    plt.close(fig)


def plot_core(rows: list[dict], fig_dir: Path) -> None:
    order = list(NAMES)
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.7), constrained_layout=True)
    panels = [
        ("reward_mean", "Mean one-second Reward"),
        ("jain_demand_satisfaction", "Jain demand-satisfaction fairness"),
        ("vo_on_time_delivery_rate", "VO timely delivery (20 ms)"),
        ("vi_on_time_delivery_rate", "VI timely delivery (50 ms)"),
    ]
    for ax, (metric, title) in zip(axes.flat, panels):
        means = [np.mean([float(r[metric]) for r in rows if r["name"] == n]) for n in order]
        ax.bar(x, means, color=[COLORS[n] for n in order], alpha=.8)
        for i, n in enumerate(order):
            vals = [float(r[metric]) for r in rows if r["name"] == n]
            ax.scatter(np.full(len(vals), i), vals, s=32, facecolor="white", edgecolor="black", zorder=3)
        ax.set_xticks(x, [n.split("_")[0] for n in order])
        ax.set_title(title)
        ax.grid(axis="y", alpha=.25)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"fixed_action_core_metrics.{ext}", dpi=220)
    plt.close(fig)


def plot_cw(cw_dir: Path, fig_dir: Path) -> None:
    with (cw_dir / "selected_action_trajectories.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), constrained_layout=True)
    styles = {"native_BEB": "-", "custom_a2_b1.6_b2.8_g6": "--"}
    for ax, ac in zip(axes.flat, ["VO", "VI", "BE", "BK"]):
        for row in rows:
            y = [int(v) for v in str(row[f"{ac}_trajectory"]).split()]
            label = f"P{row['profile']} " + ("native" if row["mode"] == "native_BEB" else "custom")
            ax.step(range(len(y)), y, where="post", lw=1.7,
                    ls=styles[row["mode"]], label=label)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("retry count")
        ax.set_ylabel("CW")
        ax.set_title(ac)
        ax.grid(True, which="both", alpha=.25)
    axes[0, 0].legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"selected_cw_trajectories.{ext}", dpi=220)
    plt.close(fig)


def plot_redundancy(cw_dir: Path, fig_dir: Path) -> None:
    with (cw_dir / "profile_redundancy.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    fig, ax1 = plt.subplots(figsize=(11.2, 5.2), constrained_layout=True)
    x = np.arange(len(rows))
    ax1.bar(x, [float(r["exact_joint_trajectories"]) for r in rows], color="#4b76b8", label="exact distinct trajectories")
    ax1.set_ylabel("exact distinct trajectories / 459 actions")
    ax1.set_xlabel("EDCA profile")
    ax1.set_xticks(x, [f"P{r['profile']}" for r in rows])
    ax2 = ax1.twinx()
    ax2.plot(x, [100 * float(r["exact_duplicate_ratio"]) for r in rows], "o-", color="#c5573f", label="exact duplicate ratio")
    ax2.set_ylabel("exact duplicate ratio (%)")
    ax2.set_ylim(0, 100)
    ax1.grid(axis="y", alpha=.25)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"cw_action_redundancy.{ext}", dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", action="append", required=True, type=Path)
    p.add_argument("--summary-dir", required=True, type=Path)
    p.add_argument("--figure-dir", required=True, type=Path)
    p.add_argument("--cw-dir", required=True, type=Path)
    args = p.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    rows, summaries = load_runs(args.run_dir)
    write_tables(rows, summaries, args.summary_dir)
    plot_ccdf(args.run_dir, args.figure_dir)
    plot_core(rows, args.figure_dir)
    plot_cw(args.cw_dir, args.figure_dir)
    plot_redundancy(args.cw_dir, args.figure_dir)
    print(json.dumps({"runs": len(rows), "seeds": sorted({int(float(r['seed'])) for r in rows}),
                      "summary_dir": str(args.summary_dir), "figure_dir": str(args.figure_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
