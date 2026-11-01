#!/usr/bin/env python3
"""Create compact diagnostics and tables for the 400-step TGax run."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config

DEFAULT_RUN = ROOT / "results/raw/tgax_ddqn_extended_seed41_400step_20261025"
SUMMARY = ROOT / "results/summary/tgax_ddqn_extended_seed41_400step_20261025"
FIGURES = ROOT / "results/figures/tgax_ddqn_extended_seed41_400step_20261025"
THRESHOLDS = ROOT / "results/summary/tgax_delay_threshold_sensitivity_20261025/threshold_sensitivity.csv"
BRANCHES = ("Profile", "alpha", "beta1", "beta2", "gamma")


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, values: list[dict]) -> None:
    if not values:
        return
    keys = []
    for row in values:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(values)


def moving(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return np.full_like(values, np.nan, dtype=float)
    out = np.full(len(values), np.nan)
    out[window - 1:] = np.convolve(values, np.ones(window) / window, mode="valid")
    return out


def threshold_plot() -> None:
    data = rows(THRESHOLDS)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    for ax, ac in zip(axes, ("VO", "VI")):
        for seed, color in ((41, "#3973b7"), (42, "#d9772b")):
            current = [r for r in data if r["ac"] == ac and int(r["seed"]) == seed]
            x = np.asarray([float(r["threshold_ms"]) for r in current])
            lo = np.asarray([float(r["timely_rate_lower_bound"]) for r in current])
            hi = np.asarray([float(r["timely_rate_upper_bound"]) for r in current])
            ax.plot(x, 100 * (lo + hi) / 2, "o-", color=color, label=f"seed {seed}")
            ax.fill_between(x, 100 * lo, 100 * hi, color=color, alpha=.18)
        ax.set(title=f"Default {ac}", xlabel="delay threshold (ms)", ylabel="timely delivery (%)",
               xticks=[20, 30, 40, 50])
        ax.grid(alpha=.25); ax.legend()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"delay_threshold_sensitivity.{ext}", dpi=220)
    plt.close(fig)


def diagnostics_plot(run: Path) -> None:
    steps, losses = rows(run / "training_steps.csv"), rows(run / "losses.csv")
    sx = np.asarray([int(r["step"]) for r in steps]); reward = np.asarray([float(r["reward"]) for r in steps])
    ux = np.asarray([int(r["update"]) for r in losses])
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), constrained_layout=True)
    ax = axes[0, 0]; ax.plot(sx, reward, alpha=.25, lw=.8); ax.plot(sx, moving(reward, 20), lw=2)
    ax.set(title="Training Reward", xlabel="environment step", ylabel="Reward")
    ax = axes[0, 1]; loss = np.asarray([float(r["td_loss"]) for r in losses]); ax.plot(ux, loss, alpha=.3)
    ax.plot(ux, moving(loss, 20), lw=2); ax.set(title="Q loss", xlabel="gradient update", ylabel="Huber loss")
    ax = axes[0, 2]
    ax.plot(ux, [float(r["q_mean"]) for r in losses], label="Online selected Q")
    ax.plot(ux, [float(r["target_q_mean"]) for r in losses], label="Target Q")
    ax.plot(ux, [float(r["td_target_mean"]) for r in losses], label="TD target", alpha=.8)
    ax.set(title="Q and TD target", xlabel="gradient update", ylabel="value"); ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(ux, [float(r["td_error_abs_mean"]) for r in losses], label="mean |TD error|")
    ax.plot(ux, [float(r["td_error_max_abs"]) for r in losses], label="max |TD error|", alpha=.7)
    ax.set(title="TD error", xlabel="gradient update", ylabel="absolute error"); ax.legend(fontsize=8)
    ax = axes[1, 1]; ax.plot(ux, [float(r["grad_norm"]) for r in losses])
    ax.axhline(10, color="r", ls="--", lw=1, label="clip norm 10")
    ax.set(title="Gradient norm before clipping", xlabel="gradient update", ylabel="norm"); ax.legend(fontsize=8)
    ax = axes[1, 2]
    for key, label in (("reward_service", "service"), ("reward_on_time_vo", "VO on-time"),
                       ("reward_on_time_vi", "VI on-time"), ("reward_retransmission", "retransmission"),
                       ("reward_low_priority_deficit", "low-priority deficit")):
        ax.plot(sx, moving(np.asarray([float(r[key]) for r in steps]), 20), label=label)
    ax.set(title="Reward components (20-step mean)", xlabel="environment step", ylabel="component")
    ax.legend(fontsize=7)
    for ax in axes.flat: ax.grid(alpha=.22)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"training_diagnostics.{ext}", dpi=220)
    plt.close(fig)


def actions_plot(run: Path, config: dict) -> None:
    data = rows(run / "training_steps.csv")
    blocks = ((1, 100), (101, 200), (201, 300), (301, 400))
    options = [list(range(16)), config["action"]["alpha"], config["action"]["beta1"],
               config["action"]["beta2"], config["action"]["gamma"]]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9.2), constrained_layout=True); axes = axes.flat
    distribution = []
    for branch, (ax, name, values) in enumerate(zip(axes, BRANCHES, options)):
        x = np.arange(len(values)); width = .19
        for bidx, (left, right) in enumerate(blocks):
            count = Counter(json.loads(r["action"])[branch] for r in data if left <= int(r["step"]) <= right)
            freq = np.asarray([count.get(i, 0) / 100 for i in range(len(values))])
            ax.bar(x + (bidx - 1.5) * width, freq, width, label=f"{left}-{right}")
            for i, value in enumerate(values):
                distribution.append({"branch": name, "block": f"{left}-{right}", "index": i,
                                     "value": value, "count": count.get(i, 0), "fraction": freq[i]})
        ax.set(title=name, ylabel="selection fraction", xticks=x,
               xticklabels=[str(v) for v in values]); ax.grid(axis="y", alpha=.2)
        if len(values) > 8: ax.tick_params(axis="x", labelrotation=45)
    axes[0].legend(fontsize=8)
    axes[-1].axis("off")
    write_csv(SUMMARY / "action_distribution_by_100_steps.csv", distribution)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"action_distribution.{ext}", dpi=220)
    plt.close(fig)


def validation_plot(run: Path) -> None:
    data = rows(run / "validation/learning_curve.csv")
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    for seed, color in ((43, "#3973b7"), (44, "#d9772b")):
        current = [r for r in data if int(r["eval_seed"]) == seed]
        x = [int(r["training_steps"]) for r in current]
        ax.plot(x, [float(r["learned_reward"]) for r in current], "o-", color=color, label=f"DDQN seed {seed}")
        ax.plot(x, [float(r["default_reward"]) for r in current], "--", color=color, alpha=.65,
                label=f"Default seed {seed}")
    ax.set(title="Independent greedy validation", xlabel="training environment steps", ylabel="mean Reward",
           xticks=[100, 200, 300, 400]); ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"independent_validation.{ext}", dpi=220)
    plt.close(fig)


def read_delay_events(path: Path) -> dict[tuple[str, str], list[float]]:
    result = {(d, a): [] for d in ("UL", "DL") for a in ("VO", "VI")}
    dmap, amap = {0: "UL", 1: "DL"}, {0: "VO", 1: "VI"}
    with (path / "events.log").open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            f = line.split()
            if f and f[0] == "Y" and 1 < float(f[1]) <= 21 and int(f[3]) in amap:
                result[(dmap[int(f[2])], amap[int(f[3])])].append(float(f[4]))
    return result


def heldout_plots(run: Path) -> None:
    base = run / "held_out/seed_45"
    policies = {"Default": base / "default", "Selected": base / "selected", "Final": base / "final"}
    colors = {"Default": "#3973b7", "Selected": "#d9772b", "Final": "#3b9955"}
    delays = {name: read_delay_events(path) for name, path in policies.items()}
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), constrained_layout=True)
    for ax, key in zip(axes.flat, [("UL", "VO"), ("DL", "VO"), ("UL", "VI"), ("DL", "VI")]):
        for name in policies:
            values = np.sort(np.asarray(delays[name][key]))
            if not len(values): continue
            ax.step(values, (len(values) - np.arange(len(values))) / len(values), where="post",
                    color=colors[name], label=name)
        ax.set_yscale("log"); ax.set_ylim(1e-4, 1.05)
        ax.set(title=f"held-out seed 45: {key[0]} {key[1]}", xlabel="MAC final service delay (ms)",
               ylabel="CCDF  P(delay > x)"); ax.grid(True, which="both", alpha=.22)
    axes[0, 0].legend()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"held_out_delay_ccdf.{ext}", dpi=220)
    plt.close(fig)

    summaries = {name: json.loads((path / "evaluation_summary.json").read_text(encoding="utf-8"))
                 for name, path in policies.items()}
    metrics = [("reward_mean", "Reward"), ("jain_demand_satisfaction", "Jain fairness"),
               ("vo_on_time_delivery_rate", "VO timely"), ("vi_on_time_delivery_rate", "VI timely")]
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.1), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, metrics):
        names = list(policies); vals = [summaries[n][metric] for n in names]
        ax.bar(names, vals, color=[colors[n] for n in names]); ax.set_title(title); ax.grid(axis="y", alpha=.22)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"held_out_core_metrics.{ext}", dpi=220)
    plt.close(fig)
    flat = []
    for name, summary in summaries.items():
        flat.append({"policy": name, **{k: v for k, v in summary.items() if not isinstance(v, (list, dict))}})
    write_csv(SUMMARY / "held_out_metrics.csv", flat)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args(); run = args.run.resolve()
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "complete":
        raise RuntimeError(f"Run is not complete: {status}")
    SUMMARY.mkdir(parents=True, exist_ok=False); FIGURES.mkdir(parents=True, exist_ok=False)
    config = load_config(ROOT / "configs/tgax_ddqn_extended_seed41.yaml")
    threshold_plot(); diagnostics_plot(run); actions_plot(run, config); validation_plot(run); heldout_plots(run)
    write_csv(SUMMARY / "validation_learning_curve.csv", rows(run / "validation/learning_curve.csv"))
    (SUMMARY / "checkpoint_selection.json").write_text(
        (run / "validation/checkpoint_selection.json").read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY), "figures": str(FIGURES)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
