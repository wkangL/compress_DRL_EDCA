"""Create the current configuration/result note and baseline CCDF figures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config
from reporting.comparison import empirical_ccdf, load_delays
from sim.traffic.payload import mean_payload
from sim.scenarios.resolver import resolve_scenario


ACS = ("vo", "vi", "be", "bk")
SHORT = {"S1_balanced_moderate": "S1 平衡中载",
         "S2_realtime_stress": "S2 实时压力",
         "S3_BE_dominant_stress": "S3 BE 主导",
         "S4_overload": "S4 过载"}
PLOT_SHORT = {"S1_balanced_moderate": "S1 Balanced moderate",
              "S2_realtime_stress": "S2 Interactive media stress",
              "S3_BE_dominant_stress": "S3 BE-dominant stress",
              "S4_overload": "S4 Overload"}


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    config = load_config(ROOT / "configs/full.yaml")
    capacity = json.loads((ROOT / config["traffic"]["capacity_file"]).read_text(encoding="utf-8"))["C_ref_mbps"]
    scenarios = load_config(ROOT / config["traffic"]["scenarios"])
    with (run / "evaluation.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(scenarios) * len(config["evaluation"]["seeds"]):
        raise ValueError("Incomplete baseline matrix")
    figure_dir = ROOT / "results/figures" / run.name
    figure_dir.mkdir(parents=True, exist_ok=True)
    for ac in ACS:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
        for ax, (name, definition) in zip(axes.flat, scenarios.items()):
            samples = [load_delays(run / f"{name}_seed{seed}")[ac]
                       for seed in config["evaluation"]["seeds"]]
            positive = [values for values in samples if len(values)]
            tail = np.concatenate([np.quantile(values, np.linspace(0, 1, min(len(values), 1500)))
                                   for values in positive])
            deepest = np.concatenate([np.sort(values)[-50:] for values in positive])
            grid = np.unique(np.r_[0, tail, deepest])
            curves = np.stack([empirical_ccdf(values, grid) for values in positive])
            center = curves.mean(axis=0); low = curves.min(axis=0); high = curves.max(axis=0)
            valid = center > 0
            ax.step(grid[valid], center[valid], where="post", color="#1565c0", label="5-seed mean")
            ax.fill_between(grid[valid], low[valid], high[valid], step="post", color="#90caf9",
                            alpha=.45, label="seed range")
            ax.set(yscale="log", title=PLOT_SHORT[name], xlabel="MAC delay (ms)", ylabel="P(D > x | ACKed)")
            ax.grid(alpha=.25); ax.legend(fontsize=8)
        fig.suptitle(f"Corrected baseline {ac.upper()} delay CCDF\n60 s × seeds 101–105; successful packets only")
        for suffix in ("png", "pdf"):
            fig.savefig(figure_dir / f"baseline_delay_ccdf_{ac}.{suffix}", dpi=180)
        plt.close(fig)
    summary_rows = []
    for name, definition in scenarios.items():
        subset = [row for row in rows if row["scenario"] == name]
        scenario=resolve_scenario(definition,config["network"]["num_sta"],capacity)
        result = {"scenario": name,"target_mbps":scenario["offered_mbps"],
                  "goodput_mbps": mean(subset, "goodput_mbps"),
                  "drop_rate": mean(subset, "drop_rate")}
        for key in ("offered_packets","overflow_count","expired_count","failed_count",
                    "other_drop_count","pending_packets_start","pending_packets_end"):
            result[key + "_mean"] = mean(subset, key)
        result["accounted_loss_fraction"]=float(np.mean([
            sum(float(row[k]) for k in ("overflow_count","expired_count","failed_count","other_drop_count"))/
            max(float(row["pending_packets_start"])+float(row["offered_packets"]),1) for row in subset]))
        for ac in ACS:
            for key in ("drop_rate", "delay_p95_ms", "delay_p99_ms", "delay_p99_9_ms", "delay_max_ms"):
                result[f"{ac}_{key}"] = mean(subset, f"{ac}_{key}")
        summary_rows.append(result)
    with (run / "summary_mean.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)
    payload_means = [mean_payload(p) for p in config["traffic"]["payload"]]
    lines = ["# 当前实验配置与修正后 baseline 结果（2026-10-19）", "",
             "本报告只把 accounting schema 2 的结果称为当前有效结果。旧 full seed 1 的 DDQN checkpoint 和图仍保留，但训练时的丢包状态与 reward 漏记溢出，不能与本表直接比较。", "",
             "## 仿真与统计配置", "",
             "- ns-3.44；1 AP、10 STA；802.11a、20 MHz；DATA 54 Mbps、控制 6 Mbps。", "- ACK 开启；RTS/CTS、隐藏节点、A-MPDU、Block Ack 关闭；不额外注入随机 PHY 错误。",
             "- retry limit=7；每设备 VO/VI/BE/BK 队列上限=100/200/500/500 包；寿命=10 s。",
             "- 包长均值 VO/VI/BE/BK=" + "/".join(f"{value:.3f}" for value in payload_means) + " bytes；VI 为 256–1500 bytes 截断对数正态。",
             "- 到达为独立泊松过程；关联预热 1 s（无业务），随后业务预热 10 s；决策周期 1 s；正式测量 60 s。",
             "- 时延为生成并调用 MAC Send 前至成功 ACK；CCDF只统计测量窗口内成功离开的包，可能包含业务预热期生成的积压包。丢弃和窗口首末在队包另列。五个 seed 等权汇总。", "",
             "## 四个流量场景", "",
             "流量改为每个 STA、每个 AC 的绝对 Mbps；C_ref 仅保留作环境状态归一化及旧实验复现，不再决定新场景负载。", "",
             "| 场景 | 每 STA VO/VI/BE/BK Mbps | 每 STA 总 Mbps | 全网目标 Mbps | 每 STA UL/DL Mbps |", "|---|---:|---:|---:|---:|"]
    for name, definition in scenarios.items():
        scenario=resolve_scenario(definition,config["network"]["num_sta"],capacity)
        target=scenario["offered_mbps"]; per_sta=target/config["network"]["num_sta"]
        ul = per_sta * definition["ul_fraction"]; dl = per_sta * (1 - definition["ul_fraction"])
        rates=definition.get("per_sta_mbps_by_ac",[per_sta*x for x in scenario["ac_mix"]])
        lines.append(f"| {SHORT[name]} | "+"/".join(f"{x:.3f}" for x in rates)+f" | {per_sta:.3f} | {target:.3f} | {ul:.3f}/{dl:.3f} |")
    lines += ["", "## 修正后 baseline：5 seeds × 60 s", "",
              "| 场景 | goodput Mbps | 丢弃/测量期到达 | 丢弃/窗口内全部待处理 | 平均溢出/过期/重试丢弃 | 期初→期末在队 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in summary_rows:
        lines.append(f"| {SHORT[row['scenario']]} | {row['goodput_mbps']:.4f} | {row['drop_rate']:.4%} | {row['accounted_loss_fraction']:.4%} | {row['overflow_count_mean']:.1f}/{row['expired_count_mean']:.1f}/{row['failed_count_mean']:.1f} | {row['pending_packets_start_mean']:.1f}→{row['pending_packets_end_mean']:.1f} |")
    lines += ["", "### 成功包时延（五个 seed 的运行级指标等权平均）", "",
              "| 场景 | VO p95 / p99 / p99.9 / max ms | VI p95 / p99.9 / max ms | BE p95 ms | BK p95 ms |",
              "|---|---:|---:|---:|---:|"]
    for row in summary_rows:
        vo = " / ".join(f"{row['vo_'+key]:.3f}" for key in ("delay_p95_ms", "delay_p99_ms", "delay_p99_9_ms", "delay_max_ms"))
        vi=" / ".join(f"{row['vi_'+key]:.3f}" for key in ("delay_p95_ms","delay_p99_9_ms","delay_max_ms"))
        lines.append(f"| {SHORT[row['scenario']]} | {vo} | {vi} | {row['be_delay_p95_ms']:.3f} | {row['bk_delay_p95_ms']:.3f} |")
    lines += ["", "### 分 AC 丢包率", "", "| 场景 | VO | VI | BE | BK |", "|---|---:|---:|---:|---:|"]
    for row in summary_rows:
        lines.append(f"| {SHORT[row['scenario']]} | {row['vo_drop_rate']:.4%} | {row['vi_drop_rate']:.4%} | {row['be_drop_rate']:.4%} | {row['bk_drop_rate']:.4%} |")
    lines += ["", "## 当前能下的结论", "",
              "- S1、S3保留原来的绝对业务速率，作为中载与BE主导对照；S2、S4提高VO/VI绝对到达率，用来形成可观测的实时业务排队尾部。",
              "- 丢包率、窗口首末积压和成功包CCDF需要合并判断；单独观察成功包时延会漏掉已经溢出、过期或仍在队列中的包。",
              "- 本轮是新绝对负载下的 baseline。DDQN 需要基于同一新配置重新训练，旧 checkpoint 不作为当前模型结果。", "",
              "## 文件", "",
              f"- 原始运行：`{run}`",
              f"- CCDF：`{figure_dir}`",
              "- 每次运行保留 events.log、periods.csv、delay_samples.npz 和 summary.json；总表为 evaluation.csv 与 summary_mean.csv。"]
    report_name=("绝对负载场景baseline结果_20261019.md" if "absolute" in run.name
                 else "当前实验配置与baseline结果_20261019.md")
    report = ROOT / "reports" / report_name
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report); print(figure_dir)


if __name__ == "__main__":
    main()
