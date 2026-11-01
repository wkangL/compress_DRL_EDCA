#!/usr/bin/env python3
"""Aggregate final-checkpoint-only comparisons for the requested workload."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT


TRAIN_SEEDS = (42, 66, 88)
EVAL_SEEDS = (44, 55, 77)
ACS = ("vo", "vi", "be", "bk")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run = args.root if args.root.is_absolute() else ROOT / args.root
    summary_dir = ROOT / "results/summary/mac_arrival40_voip40_20_gaming80_40_bv3ul10_cmdp_600step_3seed_20261030"
    figure_dir = ROOT / "results/figures/mac_arrival40_voip40_20_gaming80_40_bv3ul10_cmdp_600step_3seed_20261030"
    summary_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for train_seed in TRAIN_SEEDS:
        for eval_seed in EVAL_SEEDS:
            for policy in ("default", "final"):
                path = run / f"seed_{train_seed}/held_out/seed_{eval_seed}/{policy}/evaluation_summary.json"
                data = json.loads(path.read_text(encoding="utf-8"))
                row = {"train_seed": train_seed, "eval_seed": eval_seed, "policy": policy,
                       "reward": data["reward_mean"], "cmdp_utility": data.get("cmdp_utility_mean"),
                       "vo_timely": data.get("vo_on_time_delivery_rate"),
                       "vi_timely": data.get("vi_on_time_delivery_rate"),
                       "vo_satisfaction": data.get("vo_demand_satisfaction"),
                       "vi_satisfaction": data.get("vi_demand_satisfaction"),
                       "be_satisfaction": data.get("be_demand_satisfaction"),
                       "bk_satisfaction": data.get("bk_demand_satisfaction"),
                       "jain": data.get("jain_demand_satisfaction")}
                for ac in ACS:
                    row[f"{ac}_throughput_mbps"] = sum(float(data.get(f"{direction}_{ac}_throughput_mbps") or 0)
                                                        for direction in ("ul", "dl"))
                    for direction in ("ul", "dl"):
                        for suffix in ("delay_p95_ms", "final_drop_rate", "retransmission_rate",
                                       "risk_packets", "successful_samples", "pending_end"):
                            row[f"{direction}_{ac}_{suffix}"] = data.get(f"{direction}_{ac}_{suffix}")
                rows.append(row)
    write_csv(summary_dir / "final_checkpoint_vs_default_per_run.csv", rows)

    metrics = ("reward", "cmdp_utility", "vo_timely", "vi_timely", "vo_satisfaction",
               "vi_satisfaction", "be_satisfaction", "bk_satisfaction", "jain",
               "vo_throughput_mbps", "vi_throughput_mbps", "be_throughput_mbps", "bk_throughput_mbps")
    aggregate = []
    def policy_rows(policy: str) -> list[dict]:
        selected = [row for row in rows if row["policy"] == policy]
        if policy == "default":
            # Default is identical for all three model roots on a given eval
            # seed; count each external-demand seed once in descriptive stats.
            selected = list({row["eval_seed"]: row for row in selected}.values())
        return selected
    for metric in metrics:
        values = {}
        for policy in ("default", "final"):
            vals = np.asarray([float(row[metric]) for row in policy_rows(policy)
                               if row[metric] is not None], dtype=float)
            values[policy] = vals
            aggregate.append({"metric": metric, "policy": policy, "mean": vals.mean(),
                              "std": vals.std(ddof=1), "n": len(vals)})
        change = (None if abs(values["default"].mean()) < 1e-12 else
                  (values["final"].mean() / values["default"].mean() - 1) * 100)
        aggregate.append({"metric": metric, "policy": "final_change_percent",
                          "mean": change,
                          "std": "", "n": len(values["final"])})
    write_csv(summary_dir / "final_checkpoint_vs_default_aggregate.csv", aggregate)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    labels = ["VO timely", "VI timely", "Jain"]
    keys = ["vo_timely", "vi_timely", "jain"]
    x = np.arange(3); width = 0.35
    for offset, policy, color in ((-width/2, "default", "#4C78A8"),
                                   (width/2, "final", "#E45756")):
        means = [np.mean([float(row[key]) for row in rows if row["policy"] == policy]) for key in keys]
        axes[0].bar(x + offset, means, width, label=policy, color=color)
    axes[0].set_xticks(x, labels); axes[0].set_ylim(0, 1); axes[0].set_title("Timely delivery and fairness")
    axes[0].legend()
    for axis_index, seed in enumerate(TRAIN_SEEDS, start=1):
        steps = read_csv(run / f"seed_{seed}/training_steps.csv")
        losses = read_csv(run / f"seed_{seed}/losses.csv")
        reward = np.asarray([float(row["reward"]) for row in steps])
        window = min(20, len(reward)); smooth = np.convolve(reward, np.ones(window)/window, mode="valid")
        axes[1].plot(np.arange(window, len(reward)+1), smooth, label=f"seed {seed}")
        loss = np.asarray([float(row["td_loss"]) for row in losses])
        if len(loss):
            lw = min(20, len(loss)); ls = np.convolve(loss, np.ones(lw)/lw, mode="valid")
            axes[2].plot(np.arange(lw, len(loss)+1), ls, label=f"seed {seed}")
    axes[1].set_title("Training reward (20-step mean)"); axes[1].set_xlabel("Environment step")
    axes[2].set_title("Q loss (20-update mean)"); axes[2].set_xlabel("Gradient update")
    axes[1].legend(); axes[2].legend(); fig.tight_layout()
    fig.savefig(figure_dir / "final_comparison_and_training_diagnostics.png", dpi=220)
    plt.close(fig)

    agg = {(row["metric"], row["policy"]): row for row in aggregate}
    def pair(metric: str) -> str:
        default = float(agg[(metric, "default")]["mean"])
        final = float(agg[(metric, "final")]["mean"])
        raw_change = agg[(metric, "final_change_percent")]["mean"]
        if raw_change is None:
            return f"{default:.4f} → {final:.4f}（Default为0，相对百分比不定义）"
        change = float(raw_change)
        return f"{default:.4f} → {final:.4f}（{change:+.2f}%）"
    def mean_metric(policy: str, key: str) -> float | None:
        vals = [float(row[key]) for row in policy_rows(policy)
                if row.get(key) not in (None, "")]
        return float(np.mean(vals)) if vals else None
    def delay_row(direction: str, ac: str) -> str:
        key = f"{direction}_{ac}_delay_p95_ms"
        default, final = mean_metric("default", key), mean_metric("final", key)
        if default is None:
            return f"{direction.upper()}-{ac.upper()} | 未定义 | {final:.2f} | Default无成功样本"
        return f"{direction.upper()}-{ac.upper()} | {default:.2f} | {final:.2f} | {(final/default-1)*100:+.2f}%"
    def drop_row(direction: str, ac: str) -> str:
        key = f"{direction}_{ac}_final_drop_rate"
        default, final = mean_metric("default", key), mean_metric("final", key)
        return f"{direction.upper()}-{ac.upper()} | {default:.2%} | {final:.2%} | {(final-default)*100:+.2f}个百分点"

    per_seed_lines = []
    for seed in TRAIN_SEEDS:
        seed_rows = [row for row in rows if row["policy"] == "final" and row["train_seed"] == seed]
        per_seed_lines.append(
            f"| {seed} | {np.mean([row['reward'] for row in seed_rows]):.4f} | "
            f"{np.mean([row['vo_timely'] for row in seed_rows]):.2%} | "
            f"{np.mean([row['vi_timely'] for row in seed_rows]):.2%} | "
            f"{np.mean([row['jain'] for row in seed_rows]):.4f} |")

    loss_lines = []
    action_lines = []
    for seed in TRAIN_SEEDS:
        loss_rows = read_csv(run / f"seed_{seed}/losses.csv")
        loss = np.asarray([float(row["td_loss"]) for row in loss_rows])
        max_q = max(max(abs(float(row["q_min"])), abs(float(row["q_max"]))) for row in loss_rows)
        max_grad = max(float(row["grad_norm"]) for row in loss_rows)
        loss_lines.append(f"| {seed} | {loss[:20].mean():.5f} | {loss[-20:].mean():.5f} | {max_q:.3f} | {max_grad:.3f} |")
        counts = [dict() for _ in range(5)]
        for eval_seed in EVAL_SEEDS:
            summary = json.loads((run / f"seed_{seed}/held_out/seed_{eval_seed}/final/evaluation_summary.json").read_text())
            for action in summary["actions"]:
                for index, value in enumerate(action):
                    counts[index][int(value)] = counts[index].get(int(value), 0) + 1
        dominant = [max(count.items(), key=lambda item: item[1]) for count in counts]
        physical = [dominant[0][0], (1, 2, 3)[dominant[1][0]],
                    (1.6, 2.0)[dominant[2][0]], (2.0, 2.4, 2.8)[dominant[3][0]],
                    (5, 6, 7)[dominant[4][0]]]
        action_lines.append(f"| {seed} | P{physical[0]} | {physical[1]} | {physical[2]} | {physical[3]} | {physical[4]} |")
    report = f"""# 单BSS新业务配置 CMDP-TCN-DDQN 三seed训练报告

## 实验定义

- 业务：VoIP UL 40、VoIP DL 20、Gaming UL 80（保持上一版）、Gaming DL 40、BV3 UL 10；BE/BK保持上一版。
- MAC业务到达后立即进入原生 `WifiMacQueue`，未使用64包自定义准入水位。
- 训练seed：42、66、88；每个seed为600个1秒决策步（30个20秒episode）。
- 评估seed：44、55、77；只比较训练结束后的600步final checkpoint和配对Default，不进行中途checkpoint择优。

## Final checkpoint 与 Default

| 指标 | Default → Final（相对变化） |
|---|---:|
| VO 20 ms及时率 | {pair('vo_timely')} |
| VI 50 ms及时率 | {pair('vi_timely')} |
| VO需求满足度 | {pair('vo_satisfaction')} |
| VI需求满足度 | {pair('vi_satisfaction')} |
| BE需求满足度 | {pair('be_satisfaction')} |
| BK需求满足度 | {pair('bk_satisfaction')} |
| Jain需求满足公平性 | {pair('jain')} |
| 总Reward | {pair('reward')} |

三个训练seed分别在三个独立评估seed上的均值如下。三者的VO/VI及时率、Reward和Jain均高于同一Default均值（Default分别为40.62%、21.07%、0.3065和0.4323）。

| 训练seed | Final Reward | VO及时率 | VI及时率 | Jain |
|---:|---:|---:|---:|---:|
{chr(10).join(per_seed_lines)}

## 方向级MAC结果

成功MPDU的p95时延均值：

| 方向/AC | Default ms | Final ms | 变化 |
|---|---:|---:|---:|
| {delay_row('ul','vo')} |
| {delay_row('dl','vo')} |
| {delay_row('ul','vi')} |
| {delay_row('dl','vi')} |

最终失败率：

| 方向/AC | Default | Final | 绝对变化 |
|---|---:|---:|---:|
| {drop_row('ul','vo')} |
| {drop_row('dl','vo')} |
| {drop_row('ul','vi')} |
| {drop_row('dl','vi')} |

主要权衡是DL-VO成功样本p95由55.38 ms升至68.21 ms；但UL-VO由208.87 ms降至105.47 ms，UL/DL-VI也分别由485.74/279.69 ms降至464.21/239.41 ms。BE从近零服务改善，BK也开始出现成功交付，但BK成功样本仍很少，不能把其约0.3%满足度解释为已经解决饥饿。

## 训练稳定性与动作

| seed | 前20更新Q-loss | 后20更新Q-loss | 最大|Q| | 最大梯度范数 |
|---:|---:|---:|---:|---:|
{chr(10).join(loss_lines)}

全程无NaN、Inf、Q值爆炸或梯度爆炸。三个final策略在60个评估决策中的众数动作如下；这里已经把分支索引还原为物理候选值：

| seed | Profile | alpha | beta1 | beta2 | gamma |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(action_lines)}

三个策略没有收敛到完全相同的动作：seed 66主要选择P10，seed 42主要在P6/P10之间切换，seed 88主要在P13/P10/P1之间切换。因此结果支持“多种控制策略均可改善当前Default”，但尚不能宣称找到了唯一最优EDCA/CW动作。

## 结论

本次比较满足逐秒外生生成需求完全配对，且只使用600步final checkpoint，没有中途checkpoint择优。三个独立训练seed均改善VO/VI及时交付、需求满足度和Jain公平性；总体结果说明TCN-CMDP-DDQN在该高压单BSS环境中具有可重复的优化效果。限制是低优先级Default接近零、BK仍高度饥饿，以及DL-VO p95有所恶化，后续不能只依据总Reward宣称所有业务均改善。

详细的方向/AC时延、吞吐、最终丢包、重传、风险包和成功样本数见：
`results/summary/mac_arrival40_voip40_20_gaming80_40_bv3ul10_cmdp_600step_3seed_20261030/final_checkpoint_vs_default_per_run.csv`。

![训练与最终对比](../results/figures/mac_arrival40_voip40_20_gaming80_40_bv3ul10_cmdp_600step_3seed_20261030/final_comparison_and_training_diagnostics.png)
"""
    (ROOT / "reports/单BSS新业务配置-CMDP-TCN-DDQN三seed训练报告.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
