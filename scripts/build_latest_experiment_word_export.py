from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = "mac_arrival40_voip40_20_gaming80_40_bv3ul10_cmdp_600step_3seed_20261030"
IDENTITY = "single_bss_mac_arrival40_voip40_20_gaming80_40_bv3ul10_cmdp"
RAW = ROOT / "results" / "raw" / EXP
SUMMARY = ROOT / "results" / "summary" / EXP
FIGSRC = ROOT / "results" / "figures" / EXP
INPUTS = ROOT / "results" / "raw" / "tgax_mac_arrival40_voip40_20_gaming80_40_bv3ul10_inputs_20261030"
OUT = ROOT / "exports" / "latest_experiment_for_word"
ZIP = ROOT / "exports" / "latest_experiment_for_word.zip"


def md_table(headers, rows):
    def esc(x):
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return "未定义"
        return str(x).replace("|", "\\|").replace("\n", "<br>")
    lines = ["| " + " | ".join(map(esc, headers)) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(esc(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def fnum(v, digits=4):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "未定义"
    return f"{float(v):.{digits}f}"


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # This builder only rewrites its own fixed export tree; source results remain read-only.
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)

    status = json.loads((RAW / "status.json").read_text(encoding="utf-8"))
    cfg = json.loads((RAW / "seed_42" / "config.json").read_text(encoding="utf-8"))
    topo = pd.read_csv(INPUTS / "topology.csv")
    flows = pd.read_csv(INPUTS / "flows.csv")
    runs = pd.read_csv(SUMMARY / "final_checkpoint_vs_default_per_run.csv")
    agg = pd.read_csv(SUMMARY / "final_checkpoint_vs_default_aggregate.csv")
    profiles = json.loads((ROOT / "configs" / "profiles.json").read_text(encoding="utf-8"))

    # Default rows are intentionally repeated under each trained policy. Keep one copy per eval seed.
    default = runs[runs.policy == "default"].drop_duplicates(subset=["eval_seed"]).copy()
    final = runs[runs.policy == "final"].copy()

    # Core result tables.
    shutil.copy2(SUMMARY / "final_checkpoint_vs_default_per_run.csv", OUT / "tables" / "01_final_checkpoint_vs_default_per_run.csv")
    shutil.copy2(SUMMARY / "final_checkpoint_vs_default_aggregate.csv", OUT / "tables" / "02_final_checkpoint_vs_default_aggregate.csv")
    default.to_csv(OUT / "tables" / "03_default_unique_eval_seeds.csv", index=False, encoding="utf-8-sig")

    core = ["reward", "cmdp_utility", "vo_timely", "vi_timely", "vo_satisfaction", "vi_satisfaction",
            "be_satisfaction", "bk_satisfaction", "jain", "vo_throughput_mbps", "vi_throughput_mbps",
            "be_throughput_mbps", "bk_throughput_mbps"]
    per_train = final.groupby("train_seed")[core].mean().reset_index()
    per_train.to_csv(OUT / "tables" / "04_final_per_training_seed.csv", index=False, encoding="utf-8-sig")

    dirs = []
    for direction in ("ul", "dl"):
        for ac in ("vo", "vi", "be", "bk"):
            for metric, higher in (("delay_p95_ms", False), ("final_drop_rate", False),
                                   ("retransmission_rate", False), ("successful_samples", True),
                                   ("pending_end", False)):
                col = f"{direction}_{ac}_{metric}"
                d = pd.to_numeric(default[col], errors="coerce")
                q = pd.to_numeric(final[col], errors="coerce")
                dm, fm = d.mean(), q.mean()
                rel = (fm / dm - 1) * 100 if pd.notna(dm) and dm != 0 else np.nan
                fav = rel if higher else -rel
                dirs.append([direction.upper(), ac.upper(), metric, dm, d.std(ddof=1), len(d.dropna()),
                             fm, q.std(ddof=1), len(q.dropna()), rel, fav])
    direction_df = pd.DataFrame(dirs, columns=["direction", "ac", "metric", "default_mean", "default_std",
                                                       "default_n", "final_mean", "final_std", "final_n",
                                                       "relative_change_percent", "favorable_improvement_percent"])
    direction_df.to_csv(OUT / "tables" / "05_direction_ac_metrics_aggregate.csv", index=False, encoding="utf-8-sig")

    # Topology and traffic assignment.
    topo_out = topo.copy()
    ap = topo[topo.role == "AP"].iloc[0]
    sta = topo[topo.role == "STA"].copy()
    sta["distance_3d_m"] = np.sqrt((sta.x-ap.x)**2 + (sta.y-ap.y)**2 + (sta.z-ap.z)**2)
    topo_out = topo_out.merge(sta[["sta", "distance_3d_m"]], on="sta", how="left")
    topo_out.to_csv(OUT / "tables" / "06_topology_coordinates.csv", index=False, encoding="utf-8-sig")

    flow_counts = flows.groupby(["model", "direction", "ac"]).size().reset_index(name="flow_instances")
    flow_counts.to_csv(OUT / "tables" / "07_flow_counts.csv", index=False, encoding="utf-8-sig")
    assigned = []
    for sid in range(40):
        sub = flows[flows.sta == sid]
        counts = Counter(zip(sub.model, sub.direction, sub.ac))
        row = {"sta": sid, "total_flow_instances": len(sub)}
        for model, direction, ac in sorted(set(zip(flows.model, flows.direction, flows.ac))):
            row[f"{model}_{direction}_{ac}"] = counts[(model, direction, ac)]
        row["active_ACs"] = "/".join(sorted(sub.ac.unique()))
        assigned.append(row)
    assign_df = pd.DataFrame(assigned)
    assign_df.to_csv(OUT / "tables" / "08_traffic_assignment_per_sta.csv", index=False, encoding="utf-8-sig")

    profile_rows = []
    for i, profile in enumerate(profiles):
        for ac, p in profile.items():
            profile_rows.append([f"P{i}", ac, p["AIFSN"], p["CWmin"], p["CWmax"], p["TXOP_ms"]])
    pd.DataFrame(profile_rows, columns=["profile", "ac", "AIFSN", "CWmin", "CWmax", "TXOP_ms"]).to_csv(
        OUT / "tables" / "09_edca_profiles.csv", index=False, encoding="utf-8-sig")

    # Training diagnostics and final action distributions.
    stability = []
    action_rows = []
    vals = cfg["action"]
    names = ["profile", "alpha", "beta1", "beta2", "gamma"]
    physical = [list(range(17)), vals["alpha"], vals["beta1"], vals["beta2"], vals["gamma"]]
    for seed in status["train_seeds"]:
        loss = pd.read_csv(RAW / f"seed_{seed}" / "losses.csv")
        stability.append([seed, loss.td_loss.head(20).mean(), loss.td_loss.tail(20).mean(),
                          max(abs(loss.q_min.min()), abs(loss.q_max.max())), loss.grad_norm.max(), len(loss),
                          loss.lambda_vo.iloc[-1], loss.lambda_vi.iloc[-1], loss.lambda_jain.iloc[-1], loss.lambda_low.iloc[-1]])
        for eval_seed in cfg["evaluation"]["held_out_seeds"]:
            s = json.loads((RAW / f"seed_{seed}" / "held_out" / f"seed_{eval_seed}" / "final" / "evaluation_summary.json").read_text(encoding="utf-8"))
            for step, action in enumerate(s["actions"], start=1):
                decoded = []
                for b, idx in enumerate(action):
                    if b == 0 and idx == 16:
                        decoded.append("NativeDefault")
                    else:
                        decoded.append(physical[b][idx])
                action_rows.append([seed, eval_seed, step] + decoded)
    stability_df = pd.DataFrame(stability, columns=["train_seed", "first20_q_loss", "last20_q_loss", "max_abs_q",
                                                           "max_grad_norm", "gradient_updates", "final_lambda_vo",
                                                           "final_lambda_vi", "final_lambda_jain", "final_lambda_low"])
    stability_df.to_csv(OUT / "tables" / "10_training_stability.csv", index=False, encoding="utf-8-sig")
    action_df = pd.DataFrame(action_rows, columns=["train_seed", "eval_seed", "step", "profile", "alpha", "beta1", "beta2", "gamma"])
    action_df.to_csv(OUT / "tables" / "11_final_policy_actions.csv", index=False, encoding="utf-8-sig")
    action_dist = action_df.groupby(["train_seed", "profile", "alpha", "beta1", "beta2", "gamma"]).size().reset_index(name="count")
    action_dist.to_csv(OUT / "tables" / "12_final_action_distribution.csv", index=False, encoding="utf-8-sig")

    # Busy/queue diagnostics from unique Default and all final runs.
    diag_rows = []
    for seed in cfg["evaluation"]["held_out_seeds"]:
        d = RAW / "seed_42" / "held_out" / f"seed_{seed}" / "default" / "bss_summary.csv"
        x = pd.read_csv(d).iloc[0]
        diag_rows.append(["default", "unique", seed, x.busy_fraction, x.mean_queue_packets, x.max_queue_packets])
    for train_seed in status["train_seeds"]:
        for eval_seed in cfg["evaluation"]["held_out_seeds"]:
            d = RAW / f"seed_{train_seed}" / "held_out" / f"seed_{eval_seed}" / "final" / "bss_summary.csv"
            x = pd.read_csv(d).iloc[0]
            diag_rows.append(["final", train_seed, eval_seed, x.busy_fraction, x.mean_queue_packets, x.max_queue_packets])
    diag_df = pd.DataFrame(diag_rows, columns=["policy", "train_seed", "eval_seed", "busy_fraction", "mean_queue_packets", "max_queue_packets"])
    diag_df.to_csv(OUT / "tables" / "13_channel_queue_diagnostics.csv", index=False, encoding="utf-8-sig")

    def queue_competition(path: Path):
        # Interactive S records: time, node(0=AP,1..40=STA), AC(0..3), queue bytes, ...
        samples = defaultdict(lambda: defaultdict(dict))
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.startswith("S "):
                    continue
                p = line.split()
                if len(p) < 5:
                    continue
                t, node, ac, q = float(p[1]), int(p[2]), int(p[3]), float(p[4])
                if 1.0 <= t < 21.0 and 1 <= node <= 40:
                    samples[round(t, 6)][node][ac] = q > 0
        active, active_vo, active_vi, both = [], [], [], []
        nonempty = {ac: [] for ac in range(4)}
        for nodes in samples.values():
            active.append(sum(any(nodes[n].get(ac, False) for ac in range(4)) for n in range(1, 41)))
            active_vo.append(sum(nodes[n].get(0, False) for n in range(1, 41)))
            active_vi.append(sum(nodes[n].get(1, False) for n in range(1, 41)))
            both.append(sum(nodes[n].get(0, False) and nodes[n].get(1, False) for n in range(1, 41)))
            for ac in range(4):
                nonempty[ac].extend(nodes[n].get(ac, False) for n in range(1, 41))
        return [len(samples), np.mean(active), max(active), np.mean(active_vo), np.mean(active_vi), np.mean(both)] + [np.mean(nonempty[a]) for a in range(4)]

    competition_rows = []
    for eval_seed in cfg["evaluation"]["held_out_seeds"]:
        p = RAW / "seed_42" / "held_out" / f"seed_{eval_seed}" / "default" / "events.log"
        competition_rows.append(["default", "unique", eval_seed] + queue_competition(p))
    for train_seed in status["train_seeds"]:
        for eval_seed in cfg["evaluation"]["held_out_seeds"]:
            p = RAW / f"seed_{train_seed}" / "held_out" / f"seed_{eval_seed}" / "final" / "events.log"
            competition_rows.append(["final", train_seed, eval_seed] + queue_competition(p))
    competition_df = pd.DataFrame(competition_rows, columns=["policy", "train_seed", "eval_seed", "time_samples",
        "mean_active_sta_queues", "max_active_sta_queues", "mean_vo_nonempty_stas", "mean_vi_nonempty_stas",
        "mean_sta_both_vo_vi_nonempty", "vo_queue_nonempty_fraction", "vi_queue_nonempty_fraction",
        "be_queue_nonempty_fraction", "bk_queue_nonempty_fraction"])
    competition_df.to_csv(OUT / "tables" / "14_sta_queue_competition.csv", index=False, encoding="utf-8-sig")

    # Figures: retain the exact formal analysis figure and add two reproducible presentation figures.
    shutil.copy2(FIGSRC / "final_comparison_and_training_diagnostics.png", OUT / "figures" / "01_final_comparison_and_training_diagnostics.png")
    plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].scatter(sta.x, sta.y, s=35, color="#3478b8", alpha=.8, label="STA")
    axes[0].scatter([ap.x], [ap.y], s=180, marker="*", color="#d62728", label="AP")
    for _, r in sta.iterrows(): axes[0].text(r.x+.2, r.y+.2, str(int(r.sta)), fontsize=6)
    axes[0].set(aspect="equal", xlabel="x / m", ylabel="y / m", title="Single-BSS fixed topology (STA index)")
    axes[0].grid(alpha=.25); axes[0].legend()
    ac_order = ["VO", "VI", "BE", "BK"]
    counts = pd.crosstab(flows.sta, flows.ac).reindex(index=range(40), columns=ac_order, fill_value=0)
    bottom = np.zeros(40)
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#7f7f7f"]
    for ac, color in zip(ac_order, colors):
        axes[1].bar(range(40), counts[ac], bottom=bottom, label=ac, color=color, width=.9)
        bottom += counts[ac].to_numpy()
    axes[1].set(xlabel="STA index", ylabel="configured flow instances", title="Configured per-STA AC load")
    axes[1].legend(ncol=4); axes[1].grid(axis="y", alpha=.25)
    fig.savefig(OUT / "figures" / "02_topology_and_traffic_assignment.png", dpi=220)
    plt.close(fig)

    # Core comparison.
    def mean(policy, col):
        data = default if policy == "Default" else final
        return pd.to_numeric(data[col], errors="coerce").mean()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(5); labels = ["VO timely", "VI timely", "VO sat.", "VI sat.", "Jain"]
    cols = ["vo_timely", "vi_timely", "vo_satisfaction", "vi_satisfaction", "jain"]
    for j, pol in enumerate(["Default", "Final"]):
        axes[0,0].bar(x+(j-.5)*.34, [mean(pol,c) for c in cols], width=.34, label=pol)
    axes[0,0].set_xticks(x, labels, rotation=20); axes[0,0].set_ylim(0,1); axes[0,0].set_title("Timely delivery, demand satisfaction and fairness"); axes[0,0].legend()
    labels2=["UL-VO","DL-VO","UL-VI","DL-VI"]; cols2=["ul_vo_delay_p95_ms","dl_vo_delay_p95_ms","ul_vi_delay_p95_ms","dl_vi_delay_p95_ms"]
    for j, pol in enumerate(["Default", "Final"]): axes[0,1].bar(np.arange(4)+(j-.5)*.34,[mean(pol,c) for c in cols2],width=.34,label=pol)
    axes[0,1].set_xticks(range(4),labels2); axes[0,1].set_ylabel("ms"); axes[0,1].set_title("Successful-MPDU MAC service-delay p95")
    labels3=["VO","VI","BE","BK"]; cols3=[f"{a}_throughput_mbps" for a in ["vo","vi","be","bk"]]
    for j, pol in enumerate(["Default", "Final"]): axes[1,0].bar(np.arange(4)+(j-.5)*.34,[mean(pol,c) for c in cols3],width=.34,label=pol)
    axes[1,0].set_xticks(range(4),labels3); axes[1,0].set_ylabel("Mbps"); axes[1,0].set_yscale("symlog", linthresh=.01); axes[1,0].set_title("MAC acknowledged payload throughput (symlog)")
    labels4=["UL-VO","DL-VO","UL-VI","DL-VI"]; cols4=["ul_vo_final_drop_rate","dl_vo_final_drop_rate","ul_vi_final_drop_rate","dl_vi_final_drop_rate"]
    for j, pol in enumerate(["Default", "Final"]): axes[1,1].bar(np.arange(4)+(j-.5)*.34,[mean(pol,c) for c in cols4],width=.34,label=pol)
    axes[1,1].set_xticks(range(4),labels4); axes[1,1].set_ylim(0,.45); axes[1,1].set_title("Final failure rate")
    for ax in axes.flat: ax.grid(axis="y", alpha=.25)
    fig.savefig(OUT / "figures" / "03_core_mac_comparison.png", dpi=220)
    plt.close(fig)
    for source, target in [
        ("final_mac_service_delay_ccdf_pooled.png", "04_final_mac_service_delay_ccdf_pooled.png"),
        ("final_mac_service_delay_ccdf_by_direction.png", "05_final_mac_service_delay_ccdf_by_direction.png"),
    ]:
        if (FIGSRC / source).exists():
            shutil.copy2(FIGSRC / source, OUT / "figures" / target)

    # Reusable markdown tables.
    edca_default = md_table(["AC", "AIFSN", "CWmin", "CWmax", "TXOP"],
        [["VO",2,3,7,"2.080 ms"],["VI",2,7,15,"4.096 ms"],["BE",3,15,1023,"0"],["BK",7,15,1023,"0"]])
    traffic_table = md_table(["模型", "方向", "AC", "实例数", "MAC负载含义"],
        [[r.model, r.direction, r.ac, r.flow_instances, "随机到达，非Full Buffer"] for r in flow_counts.itertuples()])
    default_rows=[]
    for r in default.itertuples():
        total=sum(getattr(r,f"{a}_throughput_mbps") for a in ["vo","vi","be","bk"])
        default_rows.append([r.eval_seed, fnum(total,3), fnum(r.vo_throughput_mbps,3), fnum(r.vi_throughput_mbps,3),
                             fnum(r.be_throughput_mbps,4), fnum(r.bk_throughput_mbps,4), fnum(r.vo_timely), fnum(r.vi_timely), fnum(r.jain)])
    default_table=md_table(["eval seed","总吞吐/Mbps","VO","VI","BE","BK","VO及时率","VI及时率","Jain"],default_rows)
    result_rows=[]
    for metric,label,higher in [("vo_timely","VO 20 ms及时率",True),("vi_timely","VI 50 ms及时率",True),
        ("vo_satisfaction","VO需求满足度",True),("vi_satisfaction","VI需求满足度",True),
        ("be_satisfaction","BE需求满足度",True),("bk_satisfaction","BK需求满足度",True),
        ("jain","Jain需求满足公平性",True),("vo_throughput_mbps","VO吞吐/Mbps",True),
        ("vi_throughput_mbps","VI吞吐/Mbps",True),("be_throughput_mbps","BE吞吐/Mbps",True),("bk_throughput_mbps","BK吞吐/Mbps",True)]:
        dm=default[metric].mean(); fm=final[metric].mean(); rel=(fm/dm-1)*100 if dm else np.nan
        result_rows.append([label,fnum(dm),fnum(fm),fnum(rel,2)+"%" if math.isfinite(rel) else "不定义（Default=0）"])
    result_table=md_table(["指标","Default均值(n=3)","Final均值(n=9)","相对变化"],result_rows)
    delay_rows=[]
    for d,a in [("ul","vo"),("dl","vo"),("ul","vi"),("dl","vi")]:
        col=f"{d}_{a}_delay_p95_ms"; dm=default[col].mean(); fm=final[col].mean(); reduction=(dm-fm)/dm*100
        delay_rows.append([f"{d.upper()}-{a.upper()}",fnum(dm,2),fnum(fm,2),fnum(reduction,2)+"%"])
    delay_table=md_table(["方向/AC","Default p95/ms","Final p95/ms","有利改善（下降为正）"],delay_rows)
    failure_rows=[]
    for d in ("ul","dl"):
        for a in ("vo","vi","be","bk"):
            c=f"{d}_{a}_final_drop_rate"; dm=default[c].mean(); fm=final[c].mean()
            failure_rows.append([f"{d.upper()}-{a.upper()}",fnum(dm*100,2)+"%",fnum(fm*100,2)+"%",fnum((dm-fm)*100,2)+"个百分点"])
    failure_table=md_table(["方向/AC","Default最终失败率","Final最终失败率","有利绝对变化"],failure_rows)

    dist=sta.distance_3d_m
    topo_stats=f"最小 {dist.min():.2f} m、中位数 {dist.median():.2f} m、均值 {dist.mean():.2f} m、最大 {dist.max():.2f} m"

    doc0=f"""
# 00 实验身份与核实说明

## 1. 本资料包对应的唯一实验

- **正式实验标识**：`{IDENTITY}`。
- **结果目录标识**：`{EXP}`。
- **完成状态**：`status.json` 为 `complete`；结果及报告最后生成于 **2026-10-31（Asia/Shanghai）**。
- **实验类型**：单BSS、1 AP + 40 STA、MAC层随机业务直接注入、TCN-CMDP-Branching Double DQN动态EDCA/CW控制。
- **不是**：旧7 BSS TCP/UDP实验、Full Buffer实验、固定参数扫描、仅Default基线，也不是TGax Scenario 3完整复现。
- **协议路径**：运行时未安装IPv4/TCP/UDP应用链路；业务生成器产生MSDU并经 `WifiMac::Enqueue` 进入原生QoS MAC，随后仍经过EDCA、退避、TXOP、A-MPDU、Block ACK和重传。
- **训练**：训练seed 42、66、88，各600个1秒环境步；TCN从随机初始化开始并随Q学习反向传播更新，不使用预训练TCN。
- **最终评估**：每个训练seed的第600步final checkpoint分别在eval seed 44、55、77上测试；Default也在这3个eval seed上配对运行。没有中途checkpoint择优。

## 2. 对比方案

1. **Default EDCA + ns-3原生BEB**：Profile分支的独立回退动作（索引16），P0基础EDCA参数，自定义CW规则关闭。
2. **Final learned policy**：三个独立训练seed产生的最终贪心动态策略，每1秒选择一个EDCA Profile与 `(alpha,beta1,beta2,gamma)`，作用于AP及40个STA的四个AC。

## 3. 有效性核实

- 三个训练均完成30个episode、600环境步和540次梯度更新；Q-loss、Q值及梯度均为有限数，无NaN/Inf或数值爆炸。
- 评估外生业务需求按eval seed配对；同一eval seed的Default记录在三个训练目录中完全相同，汇总时去重为n=3。
- Final共有3训练seed×3评估seed=9个运行，但这9个并非9个完全独立的训练重复；统计解释需保留层级结构。
- 训练后汇总阶段曾因缺少中间validation目录而中断；600步checkpoint、Replay和评估输入保持完整，修复汇总流程后从final snapshot恢复评估，未重新训练。
- 项目当时测试结果为79 passed、7 skipped。

## 4. 尚未完全核实或需要谨慎表述

- CCA/能量检测阈值未在本实验配置中显式冻结，沿用ns-3.44 SpectrumWifiPhy默认值；本资料不猜填具体dBm。
- 没有独立隐藏节点判定实验；只能确认单AP、单信道、固定位置且无邻居BSS/外部干扰，不能据此宣称“绝无隐藏节点”。
- 静态种子阴影衰落已实现；没有另加快衰落时变模型。
- BE/BK成功样本极少，尤其Default BK为零；相关p95缺失应写“未定义”，不得写0。
- 现有结果只有3个训练seed与3个eval seed，不支持统计显著性声明或无依据置信区间。
- 工程目录不是Git工作树，因此无法提供Git commit。

## 5. 主要证据

- 运行状态：`results/raw/{EXP}/status.json`
- 每seed解析后配置：`results/raw/{EXP}/seed_42/config.json`（另外两seed配置结构相同）
- 原始输入：`{INPUTS.relative_to(ROOT).as_posix()}/topology.csv`、`flows.csv`、`manifest.json`
- 结果表：`results/summary/{EXP}/final_checkpoint_vs_default_*.csv`
- 分析脚本：`scripts/analyze_mac_arrival40_voip40_20_gaming80_40_bv3ul10_training.py`
- 正式报告：`reports/单BSS新业务配置-CMDP-TCN-DDQN三seed训练报告.md`
"""

    comp_d = competition_df[competition_df.policy == "default"]
    comp_f = competition_df[competition_df.policy == "final"]
    doc1=fr"""
# 01 仿真环境与参数配置

## 1. 平台与无线系统

| 项目 | 实际设置 | 说明 |
|---|---|---|
| 仿真器 | ns-3.44 | 本地扩展可执行程序 `ns3.44-tgax-control.exe` |
| 标准/PHY | IEEE 802.11ax HE，SpectrumWifiPhy | 单用户HE；固定速率 |
| 信道 | 5 GHz，信道号42，80 MHz | ns-3信道映射的中心频率为5210 MHz |
| 数据/控制模式 | HeMcs7 / OfdmRate6Mbps | ConstantRateWifiManager |
| 空间流 | 最大1条TX流 | 不启用OFDMA和MU-MIMO调度 |
| 保护间隔 | 800 ns | HE GI |
| 聚合 | A-MPDU启用，最大65535 B；A-MSDU=0 | Block ACK缓冲64 MPDU |
| RTS/CTS | 阈值65535 B | 普通数据帧等效不启用RTS/CTS |
| MAC队列 | 每AC最多500包，最大排队寿命500 ms | 满队列/超时会形成最终失败 |
| 重试上限 | 10 | 以本次运行时配置为准 |

STA发射功率15 dBm，Tx/Rx增益均为-2 dB，噪声系数7 dB，单天线；AP发射功率23.0103 dBm，增益0 dB，噪声系数7 dB，2根天线但最多1条TX流、2条RX流。AP信标间隔102.4 ms且启用jitter。

## 2. 传播、位置和干扰

AP固定在 `(0,0,3 m)`；40个STA固定在高度1.5 m，二维坐标见 `tables/06_topology_coordinates.csv`。AP—STA三维距离为：{topo_stats}。节点不移动。

室内路径损耗使用项目中的TGax风格分段模型：

$$L(d)=40.05+20\log_{{10}}(5/2.4)+20\log_{{10}}(\min(d,10))+\mathbf{{1}}_{{d>10}}35\log_{{10}}(d/10)+X_\sigma,$$

其中 $d$ 为米，$X_\sigma$ 为按链路和seed固定的零均值高斯阴影项，标准差5 dB。传播时延采用恒速模型。该单BSS场景没有邻居BSS或专门加入的外部干扰源；所有节点在同一80 MHz信道上竞争。CCA阈值未由实验配置显式覆盖，沿用ns-3默认值。

## 3. Default EDCA与研究动作

Default基线如下，且使用原生二进制指数退避（BEB）：失败后 $CW_r=\min(CW_{{max}},2^r(CW_{{min}}+1)-1)$，成功后复位到 $CW_{{min}}$。

{edca_default}

学习策略的Profile分支包含P0—P15（完整表在 `tables/09_edca_profiles.csv`）以及独立的Native Default回退索引16。CW参数候选为：$\alpha\in\{{1,2,3\}}$，$\beta_1\in\{{1.6,2.0\}}$，$\beta_2\in\{{2.0,2.4,2.8\}}$，$\gamma\in\{{5,6,7\}}$，并要求 $\beta_2\ge\beta_1$。16×54加1个回退动作，共865个实际合法动作。

自定义CW更新为：失败计数 $r\le\gamma$ 时，若 $r\le\alpha$ 取 $\beta=\beta_1$，否则取 $\beta=\beta_2$，然后

$$CW_r=\min\{{CW_{{max}},\lceil\beta(CW_{{r-1}}+1)\rceil-1\}}.$$

$r>\gamma$ 且尚未达到重试上限时保持CW；成功确认后复位CW和失败计数。每次动作对AP和40个STA的四个AC统一生效；单BSS中没有其他AP被控制。

## 4. MAC直达业务路径

本实验不是Full Buffer。生成器按随机业务过程产生MSDU有效载荷，加入LLC/SNAP并用QoS TID映射：VO=6、VI=5、BE=0、BK=1，然后调用原生WifiMac入队接口。`direct_high_watermark=0` 表示没有自定义64包水位或补充门控；数据到达后立即尝试进入原生MAC队列。大对象按MSDU分段，通常不超过1460 B（FTP亦有536 B分段），而不是把整个文件当作单个MAC帧。

运行时 `transport=NONE`：没有TCP、UDP、IP或Socket。`reference_transport` 仅说明原业务模型通常对应的传输层，不是本次实际协议路径。MAC层以后仍完整执行EDCA、AIFS、退避、TXOP、A-MPDU、ACK/Block ACK和重传。

## 5. 流量实例与到达模型

{traffic_table}

共246个流实例。所有40个STA均有1路VoIP UL和2路Gaming UL，因此全部STA参与VO和VI上行竞争；20路VoIP DL分配给20个STA，40路Gaming DL覆盖全部STA，BV3 UL覆盖10个STA。每个STA可同时承载多个AC，精确分配见 `tables/08_traffic_assignment_per_sta.csv`。

- **VoIP**：活跃/静默两状态，状态平均持续约1.25 s；活跃期33 B/20 ms，静默期7 B/160 ms；初态随机。DL带尺度5.11 ms的双侧随机时延扰动。
- **Gaming**：包长与包间隔分别采用项目实现的极值分布；DL包长位置/尺度390/89 B、间隔15/7 ms，UL为158/26.2 B、23.5/10.5 ms。80路UL和40路DL均保留随机到达。
- **BV3视频**：10路UL；帧大小Weibull（尺度20850 B、形状0.8099），以6 Mbps平均码率确定帧间隔，帧被拆分为MAC可承载的MSDU。
- **FTP**：文件大小截断对数正态（$\mu=14.45,\sigma=0.35$，1—5 MiB），1460 B/536 B分段，完成后指数阅读间隔均值180 s。
- **HTTP**：主对象及嵌入对象使用项目既有对数正态/截断模型，对象拆分为MSDU；页面阅读间隔指数均值30 s。

因此“配置实例数”不等于每一时刻都在MAC队列中竞争。按正式20 s内0.1 s的STA队列样本，Default平均有 **{comp_d.mean_active_sta_queues.mean():.2f}/40** 个STA至少一个AC队列非空，VO/VI分别为 {comp_d.mean_vo_nonempty_stas.mean():.2f} 和 {comp_d.mean_vi_nonempty_stas.mean():.2f} 个，同时VO与VI非空为 {comp_d.mean_sta_both_vo_vi_nonempty.mean():.2f} 个；Final对应至少一个AC非空STA均值为 {comp_f.mean_active_sta_queues.mean():.2f}。精确逐运行值见 `tables/14_sta_queue_competition.csv`。需求活跃Jain每个1秒窗口平均纳入40个STA，表示40个STA有外生需求，不等于所有40个队列每时刻都非空。该环境是高强度随机到达竞争，不应写成持续Full Buffer。
"""

    doc2=r"""
# 02 实验方案与执行过程

## 1. 研究目的

本次正式实验检验：在单BSS、40个STA同时承担高并发VO/VI随机业务并保留少量BE/BK需求时，基于STA时序观测的TCN-CMDP-Branching DDQN能否通过每秒动态修改EDCA Profile和研究型CW增长规则，提高及时交付、有效载荷交付与需求满足公平性，同时观察低优先级饥饿和方向级时延代价。

## 2. 两个正式方案

| 规范名称 | EDCA/CW | 决策方式 | 学习 |
|---|---|---|---|
| Default EDCA + Native BEB | P0标准EDCA；原生BEB；自定义CW关闭 | 20秒内固定 | 无 |
| Final TCN-CMDP-DDQN Policy | 16 Profile + CW参数或Native Default回退 | 每1秒贪心动态决策 | 每个训练seed独立训练600步 |

没有把历史固定动作、旧Reward或旧7 BSS结果混入本次比较。

## 3. 每个episode与评估运行的时间轴

1. 关联预热2 s；
2. 业务统一Default预热1 s；
3. 正式控制/测量20 s，共20个1 s决策窗口；
4. 排空1 s，用于完成或终结窗口边界数据包的账本。

一次ns-3进程覆盖约24 s仿真时间。训练每seed为30个20步episode，即600个环境步、600 s正式决策时间；若计入每episode的关联、业务预热和排空，则每seed累计约720 s ns-3仿真时间。

## 4. 训练设置

| 参数 | 数值 |
|---|---:|
| 训练seed | 42、66、88 |
| episode | 30/seed |
| 环境步 | 20/episode，600/seed |
| Replay容量/预热 | 20000 / 64 transitions |
| Batch / 每episode更新 | 32 / 20 |
| 总梯度更新 | 540/seed |
| 折扣因子 | 0.98 |
| Q网络/表示学习率 | 3e-4 / 1e-4 |
| 优化器 | AdamW |
| 梯度裁剪 | norm 10 |
| epsilon | 1.0线性降至0.05，450步 |
| Target | soft update，tau=0.001，每次更新执行 |

同一episode的20步交互期间网络参数冻结，episode结束后集中更新20次。Replay保存原始utility与cost，采样时按当前拉格朗日乘子重组奖励。TCN与STA聚合器不是固定特征提取器，而是由TD损失端到端更新。

## 5. 状态、TCN和聚合

每STA、每AC保存10个0.1 s时刻×6特征：队列字节、最终服务时延、重试、失败、丢弃、时延有效标记。共享因果TCN含3个残差块（32通道、kernel=3、dilation=1/2/4），时间池化后每AC输出8维。

Async+Age报告把8维潜变量、报告年龄和有效标记送到AP；共享MLP与注意力对40 STA逐AC聚合，每AC67维、四AC共268维。再拼接AP本地10维状态和当前动作21维（17维Profile one-hot+4个标量），最终DDQN输入为**299维**。旧文档中的298维不适用于本次新增Native Default回退后的动作编码。

Branching Q网络骨干为299→256→128，输出一个Value流和5个中心化Advantage分支，分支大小 `[17,3,2,3,3]`。Double DQN用Online网络选择下一动作、Target网络评估该动作。

## 6. CMDP效用与约束

各AC满足度 $S_a$ 为本窗口成功确认的有效载荷相对于“窗口开始尚未交付需求+本窗口新生需求”的比例，未交付需求跨窗口结转。效用变换

$$g(S)=\frac{\log(1+9S)}{\log 10},\qquad U=0.4g(S_{{VO}})+0.4g(S_{{VI}})+0.1g(S_{{BE}})+0.1g(S_{{BK}}).$$

成本为 $C_{{VO}}=1-T_{{VO}}$、$C_{{VI}}=1-T_{{VI}}$、$C_J=1-J$，以及低优先级约束成本。预算由独立Default 1 s STEP标定冻结为 `[0.61842784, 0.79366666, 0.57526697, 1.0]`；BE/BK最低满足度目标分别为 $6.1196\times10^{{-6}}$ 和0。训练即时奖励为

$$R=U-\sum_i\lambda_i\max(0,C_i-B_i),$$

其中 $B_i$ 是成本预算，$\lambda_i$ 初值0.1、学习率0.01、EMA系数0.2、上限1，每20步更新一次。重传率不直接作为CMDP成本。

## 7. 最终评估与配对

只保存并比较step 600 final snapshot；不进行中途验证或checkpoint择优。每个final策略用eval seed 44、55、77各运行20 s，Default也用相同seed和外生生成流。Final n=9来自3训练seed×3eval seed；Default去重后n=3。所有正式运行均完成，没有因性能差而排除的seed。
"""

    metric_table=md_table(["指标","单位","统计对象/方法","失败是否进入","关键注意"],[
        ["MAC成功有效载荷吞吐","Mbps","ACK/Block ACK确认的MSDU有效载荷字节×8/(1e6×20s)","失败不进分子","重传副本不重复计；A-MPDU逐MPDU确认"],
        ["最终MAC服务时延p95","ms","逐成功MPDU：首次进入MAC至最终ACK；合并20窗口原始样本后求95分位","失败不进样本","含排队、AIFS、退避、重传；无成功样本为未定义"],
        ["重传率","比例","重传尝试次数/总发送尝试次数","最终失败的尝试可进入","不是碰撞率；A-MPDU逐MPDU"],
        ["最终失败率","比例","最终失败MPDU/(测量开始待处理+测量期新入MAC MPDU)","是","队尾未完成另列，不并入失败"],
        ["及时率","比例","在生成后20ms(VO)/50ms(VI)前最终ACK的包/已到期限或提前终结的包","失败、过期计入分母","未到期限包跨窗口，不提前判失败"],
        ["需求满足度","比例","成功交付有效载荷/(结转未交付需求+新生需求)","通过未交付量体现","业务生成时刻开始，不等同MAC时延"],
        ["Jain需求满足公平性","0—1","40个有需求STA的逐STA满足比x_i；每窗口计算后取20窗口均值","零服务STA以0计入","跨所有AC/方向，不是单AC吞吐公平性"],
        ["信道忙比例","比例","PHY处于非Idle/非Sleep状态的时间占比","不适用","用于竞争诊断"],
        ["平均队列包数","packet","0.1s采样得到的节点/AC队列统计汇总","不适用","不能直接解释为40 STA同时非空"],
        ["期末未完成包","packet","排空结束仍未形成最终结果的MPDU","单独报告","不得当作成功或最终失败"],
    ])
    doc2 = doc2.replace("{{", "{").replace("}}", "}")

    doc3=r"""
# 03 评价指标定义与统计方法

## 1. 指标汇总

    @@METRIC_TABLE@@

## 2. MAC成功有效载荷吞吐量

对方向 $d$ 和AC $a$：

$$G_{{d,a}}=\frac{8B^{{ACK}}_{{d,a}}}{10^6T}\quad\text{{Mbps}},$$

其中 $B^{{ACK}}_{{d,a}}$ 是正式20 s测量窗口内最终被ACK/Block ACK确认的MAC有效载荷字节，$T=20$ s。MAC/PHY头不计入；一个MPDU的重传不会重复累计有效载荷；A-MPDU内每个MPDU按其确认结果独立计数。总吞吐为四AC、上下行成功吞吐之和。

## 3. 最终MAC服务时延

规范名称统一为“最终MAC服务时延”：

$$D_i=t^{{ACK}}_i-t^{{MAC-in}}_i.$$

$t^{{MAC-in}}_i$ 是MPDU首次被MacArrival业务送入WifiMac的时间，$t^{{ACK}}_i$ 是其最终成功ACK/Block ACK时间。因此包含MAC排队、AIFS、随机退避、TXOP等待和重传等待；不包含业务生成器在MAC之前的等待（本实验高水位为0，生成后立即尝试入MAC）。仅成功MPDU有 $D_i$；失败、过期和期末未完成包不进入时延样本，故必须同时报告失败率和未完成数。

p95由一个运行的20个正式窗口中全部逐MPDU成功样本拼接后直接求经验95分位，并非先求20个窗口p95再平均。跨seed表中的p95均值是“各运行p95的算术均值”。无成功样本返回null/“未定义”；个别底层CSV为便于写表可能显示0，不得把该0解释为零时延。

## 4. 重传率与最终失败率

$$R_{{retx}}=\frac{N_{{retransmission\ attempts}}}{N_{{all\ transmission\ attempts}}}.$$

它衡量尝试中有多少是再次发送，不能直接称为碰撞率，因为信道误码、ACK丢失等也可能触发重传。A-MPDU按MPDU跟踪尝试。

$$R_{{fail}}=\frac{N_{{retry-limit}}+N_{{queue-expire}}+N_{{enqueue-fail}}+N_{{other-final-drop}}}{N_{{pending,start}}+N_{{admitted,during}}}.$$

分母是本运行正式统计期需要服务的MPDU总体：测量开始已在MAC中的包加测量期间新入MAC的包；跨窗口挂起包只计一次。达到重试上限、500 ms队列寿命、入队失败或其他最终丢弃进入分子。排空结束仍未终结的包单列为pending_end，不进入失败分子。

## 5. 及时交付率

VO期限20 ms，VI期限50 ms，时间起点是外生业务生成时刻，终点是最终ACK：

$$T_a=\frac{N_a(t^{{ACK}}-t^{{gen}}\le L_a)}{N_a^{{eligible}}},\quad L_{{VO}}=20\text{{ ms}},\ L_{{VI}}=50\text{{ ms}}.$$

$N_a^{{eligible}}$ 包含已到期限的需求包，以及在期限前已最终失败/丢弃的包；失败、过期或到期限仍未完成均不会进入分子。窗口末尚未到期限的包保持账本并跨窗口结转，不能提前算失败。该指标比MAC服务时延多覆盖“生成到MAC入队”阶段；本实验无自定义准入水位，但仍可能发生原生队列入队失败。

## 6. 需求满足度和Jain公平性

对AC $a$，窗口可服务需求为 $A_a=D^{{carry}}_a+D^{{new}}_a$，其中 $D^{{carry}}_a$ 为上窗口未交付有效载荷，$D^{{new}}_a$ 为本窗口新生成有效载荷；成功确认有效载荷为 $Y_a$：

$$S_a=\min\left(1,\frac{Y_a}{A_a+\epsilon}\right).$$

未交付需求结转但不重复生成，因此策略不能通过减少MAC成功入队量虚增满足度。

对本窗口实际有需求的 $N$ 个STA，先计算每STA跨所有AC/方向的需求满足比 $x_i$，包括需求存在但零交付的STA（$x_i=0$），然后

$$J=\frac{(\sum_{{i=1}}^N x_i)^2}{N\sum_{{i=1}}^N x_i^2}.$$

本场景每窗口平均 $N=40$。报告值是20个1 s窗口Jain的算术均值；它不是某个AC内的吞吐公平性，也不包括没有需求的STA。

## 7. 统计边界与汇总规则

- 2 s关联预热不进入正式指标；1 s业务预热用于建立需求/队列，遗留需求会结转到正式窗口。
- 正式20个1 s窗口采用同一口径；1 s排空只用于账本闭合，不扩展吞吐测量分母。
- Default跨eval seed报告n=3；Final n=9，但同时提供按训练seed的三组均值，避免伪独立。
- 相对变化使用 $(F-D)/D\times100\%$。对“越低越好”指标，文中另给“有利改善” $(D-F)/D\times100\%$；当Default=0时百分比不定义。
""".replace("@@METRIC_TABLE@@", metric_table).replace("{{", "{").replace("}}", "}")

    pertrain_table=md_table(["训练seed","Reward","VO及时率","VI及时率","Jain"],
        [[int(r.train_seed),fnum(r.reward),fnum(r.vo_timely),fnum(r.vi_timely),fnum(r.jain)] for r in per_train.itertuples()])
    stab_table=md_table(["seed","前20 Q-loss","后20 Q-loss","max|Q|","max grad"],
        [[int(r.train_seed),fnum(r.first20_q_loss,5),fnum(r.last20_q_loss,5),fnum(r.max_abs_q,3),fnum(r.max_grad_norm,3)] for r in stability_df.itertuples()])
    busy_default=diag_df[diag_df.policy=="default"].busy_fraction.mean(); busy_final=diag_df[diag_df.policy=="final"].busy_fraction.mean()
    doc4=f"""
# 04 实验结果与分析

## 1. Default逐seed表现

{default_table}

Default总吞吐三seed均值约 **40.468 Mbps**，其中VI约40.106 Mbps、VO约0.356 Mbps，BE仅0.0050 Mbps、BK为0。信道忙比例均值为{busy_default:.3f}。这说明当前负载的主要有效服务集中在VI，BE/BK在Default下几乎完全饥饿；不能仅从总吞吐判断四AC均获得服务。

Default方向级p95：UL-VO 208.87 ms、DL-VO 55.38 ms、UL-VI 485.74 ms、DL-VI 279.69 ms。VO/VI及时率分别只有40.62%和21.07%，证实高优先级也承受显著排队/退避压力，而不是“高优先级时延总是很低”。

## 2. Final策略与Default总体对比

{result_table}

四AC吞吐相加得到Final约 **58.514 Mbps**，相对Default提高约44.59%。注意BE相对百分比巨大是因为Default分母接近0；BK Default为0，无法定义相对百分比，绝对Final也只有0.0233 Mbps。

{delay_table}

方向级最终失败率如下。重传率也总体下降，但不应把这一现象解释为“碰撞率下降”的直接证据。

{failure_table}

本实验没有输出“每AC Jain公平性”；正式Jain指标是所有有需求STA、跨AC/方向的需求满足公平性。若Word需要逐AC公平性，必须另做只读重算或后续实验，不能从现有总Jain拆分。

## 3. 三个训练seed的一致性

{pertrain_table}

三个独立策略在同一组三个eval seed上的VO/VI及时率、Reward与Jain均高于Default均值，说明改善不是单个训练seed独占。三者动作并未收敛到唯一配置：seed42主要在P6/P10之间，seed66以P10为主，seed88以P13/P10/P1为主。完整逐秒动作在 `tables/11_final_policy_actions.csv`。

## 4. 训练数值稳定性

{stab_table}

三个seed的平滑Q-loss整体低于早期，max|Q|约0.92—1.13，最大梯度范数小于0.74，未见NaN、Inf、Q值爆炸或梯度爆炸。Q-loss不要求单调下降；这里的证据只支持“数值稳定”，不等同于证明全局最优或严格收敛。

## 5. 实际观测、机制解释与假设

**实际观测结果**：Final提高VO/VI及时率、需求满足度、总吞吐和Jain；降低UL-VO、UL/ DL-VI p95及四项高优先级最终失败率。代价是DL-VO p95由55.38 ms升至68.21 ms，UL-BK失败率略恶化，BK仍高度饥饿。Final信道忙比例均值为{busy_final:.3f}。

**机制解释**：动态策略通过Profile与分段CW增长改变不同AC/节点的接入激进程度和TXOP机会。方向级重传与失败减少、VI吞吐增加与性能改善同时出现，和“减少无效竞争并提高成功聚合/服务机会”的解释一致；但现有统计不能把收益唯一归因于CW、TXOP或碰撞变化。

**待验证假设**：不同训练seed选择不同Profile却都得到改善，可能说明动作空间存在多个功能近似的局部策略；需要固定动作消融或更大规模seed才能区分Profile和CW规则各自贡献。

## 6. 局限性

- 仅3个训练seed和3个eval seed；Final的9个运行有共享训练模型的层级相关性，不能宣称统计显著。
- 单BSS且无外部干扰，不代表多BSS OBSS环境。
- MAC直达业务保留到达/包长统计但没有TCP拥塞控制、IP路由或Socket缓冲，结论限定于MAC接入优化。
- BE/BK样本稀少，低优先级时延p95容易被少数成功样本误导；核心结论应以吞吐、满足度、失败率和成功样本数联合判断。
- 及时率以业务生成时刻为起点，MAC服务时延以入MAC为起点，两者回答不同问题，不应混用。
- 结果支持“在本场景和预算下存在可重复改善”，不证明唯一最优动作，也不证明对其他拓扑/负载泛化。
"""

    doc5="""
# 05 图表目录与说明

## 图1：训练与最终性能综合诊断

- 文件：`figures/01_final_comparison_and_training_diagnostics.png`
- 来源：正式分析脚本原始输出，未重新绘制数据含义。
- 内容：训练Q-loss/Reward等诊断、final checkpoint与Default核心指标及动作分布。
- 用法：Word中可拆分说明；精确数值应引用CSV，不要从柱高读数。
- 风险：Final共有9个运行但仅3个独立训练seed；图中误差不得解释为严格置信区间。

## 图2：单BSS拓扑与业务覆盖

- 文件：`figures/02_topology_and_traffic_assignment.png`
- 左图坐标：实际x/y米坐标；星形为AP，圆点及数字为40个STA及索引。
- 右图：每STA配置的VO/VI/BE/BK流实例数（堆叠）；是配置实例，不等同于队列持续非空。
- 来源：`tables/06_topology_coordinates.csv`、`tables/08_traffic_assignment_per_sta.csv`。

## 图3：核心MAC结果对比

- 文件：`figures/03_core_mac_comparison.png`
- 左上：及时率、VO/VI需求满足度和Jain，纵轴0—1。
- 右上：成功MPDU最终MAC服务时延p95，单位ms。
- 左下：各AC成功有效载荷吞吐，单位Mbps；采用symlog便于同时显示近零BE/BK和VI。
- 右下：VO/VI方向级最终失败率。
- 风险：BE/BK近零基线造成比例视觉放大；p95仅来自成功样本。

## 图4：VO/VI合并上下行MAC服务时延CCDF

- 文件：`figures/04_final_mac_service_delay_ccdf_pooled.png`
- 横轴：成功MPDU最终MAC服务时延，单位ms；每个运行内按UL/DL成功样本数加权合并。
- 纵轴：尾概率百分比，采用对数刻度并限制在0.05%—100%，避免极少数最尾端样本主导视觉判断。
- 曲线：Default为3个唯一eval seed的逐运行CCDF均值；Final-42/66/88分别是对应训练模型在3个eval seed上的均值。浅色区域为这3个eval seed的最小—最大范围，不是置信区间。
- 注意：仅包含成功确认MPDU；最终失败和期末未完成包必须结合失败率表解释。VO 20 ms、VI 50 ms及时率以业务生成时刻为起点，与本图MAC入队起点不同，不能直接从本图读取。

## 图5：按上下行区分的VO/VI MAC服务时延CCDF

- 文件：`figures/05_final_mac_service_delay_ccdf_by_direction.png`
- 四个面板依次为UL-VO、DL-VO、UL-VI、DL-VI，坐标及曲线汇总规则与图4相同。
- 用途：展示合并曲线背后的方向差异，特别是Final改善UL-VO而DL-VO p95有所恶化的权衡。

## 原始结果表

| 文件 | 用途 |
|---|---|
| `01_final_checkpoint_vs_default_per_run.csv` | 全部正式运行、方向/AC指标与样本数 |
| `02_final_checkpoint_vs_default_aggregate.csv` | 原分析脚本汇总均值、标准差、相对变化 |
| `03_default_unique_eval_seeds.csv` | 去除重复Default后的3个eval seed |
| `04_final_per_training_seed.csv` | 每个独立训练seed在3个eval seed上的均值 |
| `05_direction_ac_metrics_aggregate.csv` | 方向级p95、失败、重传、样本数及有利改善 |
| `06_topology_coordinates.csv` | AP/STA坐标与AP距离 |
| `07_flow_counts.csv` | 业务模型/方向/AC实例数 |
| `08_traffic_assignment_per_sta.csv` | 40 STA精确业务分配 |
| `09_edca_profiles.csv` | P0—P15四AC参数 |
| `10_training_stability.csv` | Q-loss、Q值、梯度与最终拉格朗日乘子 |
| `11_final_policy_actions.csv` | 9个final评估每秒动作 |
| `12_final_action_distribution.csv` | 动作组合计数 |
| `13_channel_queue_diagnostics.csv` | 信道忙比例和队列摘要 |
| `14_sta_queue_competition.csv` | 0.1 s采样的STA队列非空比例与同时竞争STA数 |
"""

    for name, text in [("00_实验身份与核实说明.md",doc0),("01_仿真环境与参数配置.md",doc1),
                       ("02_实验方案与执行过程.md",doc2),("03_评价指标定义与统计方法.md",doc3),
                       ("04_实验结果与分析.md",doc4),("05_图表目录与说明.md",doc5)]:
        write_text(OUT / name, text)

    # Manifest after all deliverables exist.
    git_commit="不可用：D:/compress_DRL_EDCA 不是Git工作树"
    manifest=f"""
# 交付清单

## 实验元数据

- 配置标识：`{IDENTITY}`
- 结果标识：`{EXP}`
- 完成时间：2026-10-31（Asia/Shanghai；依据status与汇总文件时间）
- Git commit：{git_commit}
- 原始结果目录：`results/raw/{EXP}`
- 汇总目录：`results/summary/{EXP}`
- 图目录：`results/figures/{EXP}`
- 分析脚本：`scripts/analyze_mac_arrival40_voip40_20_gaming80_40_bv3ul10_training.py`
- 本资料包生成脚本：`scripts/build_latest_experiment_word_export.py`

## 文档

- `00_实验身份与核实说明.md`：唯一实验身份、证据、有效性和不确定项。
- `01_仿真环境与参数配置.md`：拓扑、PHY/MAC、传播、EDCA/CW和业务模型。
- `02_实验方案与执行过程.md`：目标、方案、时间轴、状态/网络、CMDP与评估设计。
- `03_评价指标定义与统计方法.md`：公式、分母、窗口和缺失样本处理。
- `04_实验结果与分析.md`：Default、Final、逐seed、权衡、机制解释和限制。
- `05_图表目录与说明.md`：Word插图与CSV索引。

## figures 与 tables

`figures/`仅保留本实验正式图及由本实验输入/正式CSV重绘的四张说明图；`tables/`包含14张可复核CSV。未包含checkpoint、Replay、完整events.log或历史实验。

## 完整性校验

以下SHA-256在生成时计算；压缩包建立后另以 `zipfile.testzip()` 检查CRC。
"""
    rows=[]
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "manifest.md":
            rows.append([p.relative_to(OUT).as_posix(),p.stat().st_size,sha256(p)])
    manifest += "\n\n"+md_table(["文件","字节","SHA-256"],rows)
    write_text(OUT / "manifest.md", manifest)

    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file(): z.write(p, Path("latest_experiment_for_word") / p.relative_to(OUT))
    with zipfile.ZipFile(ZIP, "r") as z:
        bad=z.testzip()
        if bad: raise RuntimeError(f"ZIP CRC failure: {bad}")
    print(json.dumps({"export":str(OUT),"zip":str(ZIP),"files":sum(p.is_file() for p in OUT.rglob('*')),
                      "zip_bytes":ZIP.stat().st_size},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
