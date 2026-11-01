from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = "mac_arrival40_voip40_20_gaming80_40_bv3ul10_cmdp_600step_3seed_20261030"
RAW = ROOT / "results" / "raw" / EXP
OUT = ROOT / "results" / "figures" / EXP
X = np.linspace(0.0, 500.0, 2001)
TRAIN_SEEDS = (42, 66, 88)
EVAL_SEEDS = (44, 55, 77)
COLORS = {"Default": "#222222", "Final-42": "#1f77b4", "Final-66": "#ff7f0e", "Final-88": "#2ca02c"}


def run_dir(train_seed: int, eval_seed: int, policy: str) -> Path:
    return RAW / f"seed_{train_seed}" / "held_out" / f"seed_{eval_seed}" / policy


def one_curve(directory: Path, ac: str, direction: str | None) -> np.ndarray:
    frame = pd.read_csv(directory / "mac_service_delay_ccdf.csv")
    frame = frame[(frame["scope"] == "BSS") & (frame["ac"] == ac)]
    summary = json.loads((directory / "evaluation_summary.json").read_text(encoding="utf-8"))
    directions = (direction,) if direction else ("UL", "DL")
    weighted = np.zeros_like(X)
    total = 0
    for d in directions:
        part = frame[frame["direction"] == d].sort_values("delay_ms")
        n = int(summary[f"{d.lower()}_{ac.lower()}_successful_samples"])
        if n == 0 or part.empty:
            continue
        values = np.interp(X, part.delay_ms.to_numpy(), part.ccdf.to_numpy(), left=1.0, right=0.0)
        weighted += n * values
        total += n
    return weighted / total if total else np.full_like(X, np.nan)


def policy_curves(ac: str, direction: str | None):
    result: dict[str, list[np.ndarray]] = {name: [] for name in COLORS}
    # Default is identical under every training directory, so use one copy per eval seed.
    for e in EVAL_SEEDS:
        result["Default"].append(one_curve(run_dir(42, e, "default"), ac, direction))
    for t in TRAIN_SEEDS:
        for e in EVAL_SEEDS:
            result[f"Final-{t}"].append(one_curve(run_dir(t, e, "final"), ac, direction))
    return result


def draw_axis(ax, ac: str, direction: str | None):
    for name, curves in policy_curves(ac, direction).items():
        arr = np.asarray(curves)
        valid = arr[~np.isnan(arr).all(axis=1)]
        if not len(valid):
            continue
        mean = np.nanmean(valid, axis=0) * 100.0
        ax.plot(X, mean, lw=2.2, color=COLORS[name], label=name)
        if len(valid) > 1:
            lo, hi = np.nanmin(valid, axis=0) * 100.0, np.nanmax(valid, axis=0) * 100.0
            ax.fill_between(X, np.maximum(lo, 0.05), np.maximum(hi, 0.05), color=COLORS[name], alpha=0.07)
    ax.set_yscale("log")
    ax.set_xlim(0, 500)
    ax.set_ylim(0.05, 100)
    ax.set_yticks([100, 50, 10, 1, 0.1, 0.05], ["100", "50", "10", "1", "0.1", "0.05"])
    title = f"{ac} pooled UL+DL" if direction is None else f"{direction}-{ac}"
    ax.set_title(title)
    ax.set_xlabel("successful MPDU MAC service delay / ms")
    ax.set_ylabel("tail probability / %")
    ax.grid(True, which="both", alpha=0.22)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    draw_axis(axes[0], "VO", None)
    draw_axis(axes[1], "VI", None)
    axes[0].legend(loc="lower left")
    fig.suptitle("Final MAC service-delay CCDF (UL and DL pooled within each run)")
    fig.savefig(OUT / "final_mac_service_delay_ccdf_pooled.png", dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for ax, (direction, ac) in zip(axes.flat, (("UL", "VO"), ("DL", "VO"), ("UL", "VI"), ("DL", "VI"))):
        draw_axis(ax, ac, direction)
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("Final MAC service-delay CCDF by direction and AC")
    fig.savefig(OUT / "final_mac_service_delay_ccdf_by_direction.png", dpi=240)
    plt.close(fig)

    print(OUT / "final_mac_service_delay_ccdf_pooled.png")
    print(OUT / "final_mac_service_delay_ccdf_by_direction.png")


if __name__ == "__main__":
    main()
