#!/usr/bin/env python3
"""Create the compact CSV/PNG deliverables for the completed Reward-v3 run."""
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

DEFAULT_RUN = ROOT / "results/raw/tgax_ddqn_reward_v3_seed41_400step_20261027"
SUMMARY = ROOT / "results/summary/tgax_ddqn_reward_v3_seed41_400step_20261028"
FIGURES = ROOT / "results/figures/tgax_ddqn_reward_v3_seed41_400step_20261028"
ACS = ("vo", "vi", "be", "bk")
BRANCHES = ("Profile", "alpha", "beta1", "beta2", "gamma")

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def moving(values: np.ndarray, window: int = 20) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(values) >= window:
        out[window - 1:] = np.convolve(values, np.ones(window) / window, mode="valid")
    return out


def training_plot(run: Path) -> dict:
    steps, losses = read_csv(run / "training_steps.csv"), read_csv(run / "losses.csv")
    sx = np.asarray([int(r["step"]) for r in steps]); ux = np.asarray([int(r["update"]) for r in losses])
    reward = np.asarray([float(r["reward"]) for r in steps]); qloss = np.asarray([float(r["td_loss"]) for r in losses])
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), constrained_layout=True)
    axes[0, 0].plot(sx, reward, alpha=.22, lw=.7); axes[0, 0].plot(sx, moving(reward), lw=2)
    axes[0, 0].set(title="训练Reward-v3", xlabel="环境步", ylabel="Reward")
    axes[0, 1].plot(ux, qloss, alpha=.22, lw=.7); axes[0, 1].plot(ux, moving(qloss), lw=2)
    axes[0, 1].set(title="Q-loss", xlabel="梯度更新", ylabel="Huber TD loss")
    axes[0, 2].plot(ux, [float(r["q_mean"]) for r in losses], label="Online Q")
    axes[0, 2].plot(ux, [float(r["target_q_mean"]) for r in losses], label="Target Q")
    axes[0, 2].plot(ux, [float(r["td_target_mean"]) for r in losses], label="TD target", alpha=.8)
    axes[0, 2].set(title="Q值与TD目标", xlabel="梯度更新"); axes[0, 2].legend(fontsize=8)
    axes[1, 0].plot(ux, [float(r["td_error_abs_mean"]) for r in losses], label="mean |TD|")
    axes[1, 0].plot(ux, [float(r["td_error_max_abs"]) for r in losses], label="max |TD|", alpha=.7)
    axes[1, 0].set(title="TD误差", xlabel="梯度更新"); axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(ux, [float(r["grad_norm"]) for r in losses]); axes[1, 1].axhline(10, color="r", ls="--", lw=1)
    axes[1, 1].set(title="梯度范数（裁剪阈值10）", xlabel="梯度更新")
    for key, label in (("reward_on_time_vo", "VO及时率"), ("reward_on_time_vi", "VI及时率"),
                       ("reward_service_all", "需求满足"), ("reward_jain", "Jain")):
        axes[1, 2].plot(sx, moving(np.asarray([float(r[key]) for r in steps])), label=label)
    axes[1, 2].set(title="Reward-v3组成（20步均值）", xlabel="环境步"); axes[1, 2].legend(fontsize=8)
    for ax in axes.flat: ax.grid(alpha=.22)
    fig.savefig(FIGURES / "training_diagnostics.png", dpi=220); plt.close(fig)
    fields = ("td_loss", "q_min", "q_max", "target_q_min", "target_q_max", "td_error_max_abs", "grad_norm")
    finite = all(math.isfinite(float(r[k])) for r in losses for k in fields)
    return {"environment_steps": len(steps), "gradient_updates": len(losses), "all_finite": finite,
            "qloss_first20_mean": float(qloss[:20].mean()), "qloss_last20_mean": float(qloss[-20:].mean()),
            "qloss_max": float(qloss.max()), "q_abs_max": max(abs(float(r[k])) for r in losses
                for k in ("q_min", "q_max", "target_q_min", "target_q_max")),
            "gradient_norm_max": max(float(r["grad_norm"]) for r in losses)}


def validation_plot(run: Path) -> None:
    data = read_csv(run / "validation/learning_curve.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.3), constrained_layout=True)
    metrics = (("reward", "Reward-v3"), ("vo_on_time_delivery_rate", "VO 20 ms及时率"),
               ("vi_on_time_delivery_rate", "VI 50 ms及时率"),
               ("jain_demand_satisfaction", "Jain需求满足公平性"))
    for ax, (metric, title) in zip(axes.flat, metrics):
        for seed, color in ((44, "#3973b7"), (45, "#d9772b")):
            current = [r for r in data if int(r["eval_seed"]) == seed]
            x = [int(r["training_steps"]) for r in current]
            learned_key = "learned_reward" if metric == "reward" else f"learned_{metric}"
            default_key = "default_reward" if metric == "reward" else None
            ax.plot(x, [float(r[learned_key]) for r in current], "o-", color=color, label=f"DDQN seed {seed}")
            if default_key:
                ax.plot(x, [float(r[default_key]) for r in current], "--", color=color, alpha=.65,
                        label=f"Default seed {seed}")
            else:
                base_path = run / "validation/baseline" / f"seed_{seed}" / "evaluation_summary.json"
                base = json.loads(base_path.read_text(encoding="utf-8"))[metric]
                ax.axhline(base, ls="--", color=color, alpha=.65)
        ax.set(title=title, xlabel="训练环境步", xticks=[100, 200, 300, 400]); ax.grid(alpha=.22)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.savefig(FIGURES / "independent_validation.png", dpi=220); plt.close(fig)
    write_csv(SUMMARY / "validation_learning_curve.csv", data)


def action_plot(run: Path, config: dict) -> None:
    data = read_csv(run / "training_steps.csv")
    blocks = ((1, 100), (101, 200), (201, 300), (301, 400))
    options = [list(range(17)), config["action"]["alpha"], config["action"]["beta1"],
               config["action"]["beta2"], config["action"]["gamma"]]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9.2), constrained_layout=True); flat = axes.flat
    output = []
    for branch, (ax, name, values) in enumerate(zip(flat, BRANCHES, options)):
        x = np.arange(len(values)); width = .19
        for bidx, (left, right) in enumerate(blocks):
            count = Counter(json.loads(r["action"])[branch] for r in data if left <= int(r["step"]) <= right)
            fraction = np.asarray([count.get(i, 0) / 100 for i in range(len(values))])
            ax.bar(x + (bidx - 1.5) * width, fraction, width, label=f"{left}-{right}")
            for i, value in enumerate(values):
                output.append({"branch": name, "block": f"{left}-{right}", "index": i,
                               "value": "Default fallback" if name == "Profile" and i == 16 else value,
                               "count": count.get(i, 0), "fraction": fraction[i]})
        ax.set(title=name, ylabel="选择比例", xticks=x, xticklabels=[str(v) for v in values]); ax.grid(axis="y", alpha=.2)
        if len(values) > 8: ax.tick_params(axis="x", labelrotation=45)
    axes[0, 0].legend(fontsize=8); axes[-1, -1].axis("off")
    fig.savefig(FIGURES / "training_action_distribution.png", dpi=220); plt.close(fig)
    write_csv(SUMMARY / "training_action_distribution.csv", output)

    greedy = []
    for step in (100, 200, 300, 400):
        for seed in (44, 45):
            summary = json.loads((run / "validation" / f"step_{step}" / f"seed_{seed}" /
                                  "evaluation_summary.json").read_text(encoding="utf-8"))
            for action, count in Counter(tuple(x) for x in summary["actions"]).items():
                greedy.append({"checkpoint_step": step, "eval_seed": seed, "action": json.dumps(action),
                               "profile": action[0], "alpha_index": action[1], "beta1_index": action[2],
                               "beta2_index": action[3], "gamma_index": action[4], "count": count})
    write_csv(SUMMARY / "greedy_validation_actions.csv", greedy)


def exact_rates(directory: Path) -> tuple[float, float]:
    steps = read_csv(directory / "evaluation_steps.csv")
    failed = risk = retries = attempts = 0
    for ac in ACS:
        for direction in ("ul", "dl"):
            p = f"{direction}_{ac}"
            failed += sum(int(float(r[f"{p}_failed"])) for r in steps)
            risk += int(float(steps[0][f"{p}_pending_start"])) + sum(
                int(float(r[f"{p}_arrived"])) for r in steps)
            retries += sum(int(float(r[f"{p}_retransmissions"])) for r in steps)
            attempts += sum(int(float(r[f"{p}_tx_attempts"])) for r in steps)
    return (failed / risk if risk else float("nan"), retries / attempts if attempts else float("nan"))


def aggregate_delay_p95(directory: Path) -> dict[str, float | None]:
    values = {ac: [] for ac in ACS}
    with (directory / "events.log").open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            fields = line.split()
            if fields and fields[0] == "Y" and 1.0 < float(fields[1]) <= 21.0:
                ac = ACS[int(float(fields[3]))]
                values[ac].append(float(fields[4]))
    return {ac: float(np.percentile(samples, 95)) if samples else None for ac, samples in values.items()}


def heldout(run: Path) -> list[dict]:
    rows = []
    for seed in (46, 47):
        for policy in ("default", "selected", "final"):
            directory = run / "held_out" / f"seed_{seed}" / policy
            summary = json.loads((directory /
                                  "evaluation_summary.json").read_text(encoding="utf-8"))
            generated = np.asarray(summary["application_generated_bytes"], dtype=float)
            received = np.asarray(summary["application_received_bytes"], dtype=float)
            failure, retry = exact_rates(directory)
            aggregate_p95 = aggregate_delay_p95(directory)
            row = {"eval_seed": seed, "policy": policy,
                   "checkpoint_step": 0 if policy == "default" else (100 if policy == "selected" else 400),
                   "reward_v3": summary["reward_mean"], "vo_timely": summary["vo_on_time_delivery_rate"],
                   "vi_timely": summary["vi_on_time_delivery_rate"],
                   "jain": summary["jain_demand_satisfaction"],
                   "online_jain_mean": summary["online_jain_demand_satisfaction_mean"],
                   "application_satisfaction_total": float(received.sum() / generated.sum()),
                   "final_failure_rate": failure, "retransmission_rate": retry}
            for index, ac in enumerate(ACS):
                row[f"{ac}_application_satisfaction"] = float(received[index] / generated[index]) if generated[index] else None
                row[f"{ac}_throughput_mbps"] = sum(float(summary.get(f"{d}_{ac}_throughput_mbps") or 0.) for d in ("ul", "dl"))
                samples = [summary.get(f"{d}_{ac}_delay_p95_ms") for d in ("ul", "dl")]
                valid = [float(x) for x in samples if x is not None]
                row[f"{ac}_worst_direction_p95_ms"] = max(valid) if valid else None
                row[f"{ac}_aggregate_p95_ms"] = aggregate_p95[ac]
            rows.append(row)
    for seed in (46, 47):
        base = next(r for r in rows if r["eval_seed"] == seed and r["policy"] == "default")
        for row in (r for r in rows if r["eval_seed"] == seed):
            for metric in ("reward_v3", "vo_timely", "vi_timely", "jain", "application_satisfaction_total",
                           "vo_throughput_mbps", "vi_throughput_mbps", "be_throughput_mbps", "bk_throughput_mbps"):
                b = float(base[metric]); row[f"{metric}_change_percent"] = (float(row[metric]) / b - 1) * 100 if b else None
    write_csv(SUMMARY / "heldout_core_metrics.csv", rows)
    means = {p: {m: float(np.mean([r[m] for r in rows if r["policy"] == p]))
                 for m in ("reward_v3", "vo_timely", "vi_timely", "jain", "application_satisfaction_total")}
             for p in ("default", "selected", "final")}
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.1), constrained_layout=True)
    metrics = (("reward_v3", "Reward-v3"), ("vo_timely", "VO 20 ms及时率"),
               ("vi_timely", "VI 50 ms及时率"), ("jain", "Jain公平性"),
               ("application_satisfaction_total", "应用有效载荷/生成需求"))
    colors = ["#3973b7", "#d9772b", "#3b9955"]
    for ax, (metric, title) in zip(axes.flat, metrics):
        ax.bar(("Default", "Selected-100", "Final-400"),
               [means[p][metric] for p in ("default", "selected", "final")], color=colors)
        ax.set_title(f"留出seed 46/47均值：{title}"); ax.grid(axis="y", alpha=.22); ax.tick_params(axis="x", rotation=12)
    axes[-1, -1].axis("off")
    fig.savefig(FIGURES / "heldout_core_metrics.png", dpi=220); plt.close(fig)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args(); run = args.run.resolve()
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "complete": raise RuntimeError(f"Run incomplete: {status}")
    SUMMARY.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "configs/tgax_ddqn_reward_v3.yaml")
    diagnostics = training_plot(run); validation_plot(run); action_plot(run, config); held = heldout(run)
    selection = json.loads((run / "validation/checkpoint_selection.json").read_text(encoding="utf-8"))
    manifest = {"run": str(run), "summary": str(SUMMARY), "figures": str(FIGURES),
                "diagnostics": diagnostics, "checkpoint_selection": selection,
                "heldout_rows": len(held)}
    (SUMMARY / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
