#!/usr/bin/env python3
"""Create clearly-labelled live training diagnostics for the v6 multi-seed run."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


def rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def smooth(values: np.ndarray, window: int = 20) -> np.ndarray:
    if len(values) < window:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full(len(values), np.nan)
    result[window - 1:] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    loss_fig, loss_ax = plt.subplots(figsize=(9.2, 5.2))
    reward_fig, reward_ax = plt.subplots(figsize=(9.2, 5.2))
    for root in args.roots:
        seed = int(root.name.split("seed", 1)[1].split("_", 1)[0])
        losses = rows(root / "losses.csv")
        steps = rows(root / "training_steps.csv")
        updates = np.asarray([int(row["update"]) for row in losses])
        qloss = np.asarray([float(row["td_loss"]) for row in losses])
        env_steps = np.asarray([int(row["step"]) for row in steps])
        reward = np.asarray([float(row["reward"]) for row in steps])
        if len(qloss):
            loss_ax.plot(updates, qloss, alpha=.16, linewidth=.7)
            loss_ax.plot(updates, smooth(qloss), linewidth=2, label=f"seed {seed}（20更新滑动均值）")
        if len(reward):
            reward_ax.plot(env_steps, reward, alpha=.13, linewidth=.7)
            reward_ax.plot(env_steps, smooth(reward), linewidth=2, label=f"seed {seed}（20步滑动均值）")
        validation = rows(root / "validation" / "learning_curve.csv")
        summary.append({
            "seed": seed, "environment_steps_written": len(steps),
            "gradient_updates_written": len(losses),
            "qloss_first20_mean": float(qloss[:20].mean()) if len(qloss) >= 20 else None,
            "qloss_last20_mean": float(qloss[-20:].mean()) if len(qloss) >= 20 else None,
            "qloss_max": float(qloss.max()) if len(qloss) else None,
            "q_min": min((float(row["q_min"]) for row in losses), default=None),
            "q_max": max((float(row["q_max"]) for row in losses), default=None),
            "gradient_norm_max": max((float(row["grad_norm"]) for row in losses), default=None),
            "reward_first50_mean": float(reward[:50].mean()) if len(reward) >= 50 else None,
            "reward_last50_mean": float(reward[-50:].mean()) if len(reward) >= 50 else None,
            "validation_rows": len(validation),
        })
    loss_ax.set(xlabel="梯度更新次数", ylabel="Huber TD loss",
                title="TCN-DDQN Q-loss（训练中快照，不代表MAC性能）")
    loss_ax.grid(alpha=.25); loss_ax.legend(); loss_fig.tight_layout()
    loss_fig.savefig(args.output / "live_qloss.png", dpi=220); plt.close(loss_fig)
    reward_ax.set(xlabel="环境步", ylabel="Reward",
                  title="TCN-DDQN训练Reward（训练中快照）")
    reward_ax.grid(alpha=.25); reward_ax.legend(); reward_fig.tight_layout()
    reward_fig.savefig(args.output / "live_reward.png", dpi=220); plt.close(reward_fig)
    (args.output / "live_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
