"""Summarize single-AC load calibration and queue sensitivity."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT
from reporting.comparison import empirical_ccdf


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def avg(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def group(rows: list[dict], ac: str, rate: float) -> list[dict]:
    return [row for row in rows if row["ac"] == ac and float(row["per_sta_target_mbps"]) == rate]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan", type=Path)
    parser.add_argument("sensitivity", type=Path)
    args = parser.parse_args()
    scan = args.scan.resolve(); sensitivity = args.sensitivity.resolve()
    rows = read(scan / "runs.csv"); large = read(sensitivity / "runs.csv")
    out = ROOT / "results/figures" / scan.name; out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for column, ac in enumerate(("vo", "vi")):
        rates = sorted({float(row["per_sta_target_mbps"]) for row in rows if row["ac"] == ac})
        p95 = [avg(group(rows, ac, rate), "delay_p95_ms") for rate in rates]
        p999 = [avg(group(rows, ac, rate), "delay_p99_9_ms") for rate in rates]
        drops = [avg(group(rows, ac, rate), "drop_rate") for rate in rates]
        pending = [avg(group(rows, ac, rate), "pending_mean") for rate in rates]
        axes[0, column].plot(rates, p95, "o-", label="p95")
        axes[0, column].plot(rates, p999, "s-", label="p99.9")
        axes[0, column].set(yscale="log", title=f"{ac.upper()} successful-packet delay",
                            xlabel="Offered Mbps per STA", ylabel="Delay (ms)")
        axes[0, column].grid(alpha=.25); axes[0, column].legend()
        axis = axes[1, column]
        axis.plot(rates, np.asarray(drops) * 100, "o-", color="#c62828", label="drop rate")
        axis.set(title=f"{ac.upper()} loss and backlog", xlabel="Offered Mbps per STA",
                 ylabel="Drop rate (%)")
        axis.set_yscale("symlog", linthresh=.01)
        twin = axis.twinx(); twin.plot(rates, pending, "s--", color="#1565c0", label="pending packets")
        twin.set_ylabel("Mean pending packets")
        axis.grid(alpha=.25)
        handles = axis.lines + twin.lines
        axis.legend(handles, [line.get_label() for line in handles], loc="upper left")
    fig.suptitle("Single-AC load calibration: 10 STA, 50/50 UL/DL\n10 s traffic warmup + 20 s measurement; 3-seed mean")
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"ac_load_response.{suffix}", dpi=180)
    plt.close(fig)
    # Successful-packet CCDF around each onset point.
    selected = {"vo": (0.75, 1.00, 1.50), "vi": (2.00, 2.50, 3.00)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for axis, ac in zip(axes, ("vo", "vi")):
        for rate in selected[ac]:
            samples = []
            for seed in (501, 502, 503):
                token = str(rate).replace(".", "p")
                with np.load(scan / f"{ac}_per_sta_{token}mbps_seed{seed}" / "delay_samples.npz") as packed:
                    samples.append(np.asarray(packed[ac]))
            grid = np.unique(np.concatenate([np.quantile(values, np.linspace(0, 1, min(len(values), 2000)))
                                             for values in samples] + [np.sort(values)[-50:] for values in samples]))
            curve = np.mean([empirical_ccdf(values, grid) for values in samples], axis=0)
            valid = curve > 0; axis.step(grid[valid], curve[valid], where="post", label=f"{rate:.2f} Mbps/STA")
        axis.set(yscale="log", title=ac.upper(), xlabel="MAC delay (ms)", ylabel="P(D > x | ACKed)")
        axis.grid(alpha=.25); axis.legend()
    fig.suptitle("Successful-packet CCDF near the queueing onset")
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"ac_onset_ccdf.{suffix}", dpi=180)
    plt.close(fig)
    lines = ["# VO/VI 绝对负载标定与队列容量敏感性（2026-10-19）", "",
             "固定 10 STA、上下行各 50%、默认 EDCA，主配置寿命 10 秒、retry limit 7。每点 seeds 501–503，带业务预热 10 秒后测量 20 秒，队列不中途清空。", "",
             "## 单 AC 扫描（三 seed 均值）", "",
             "| AC | 每 STA Mbps | 实测全网 offered/goodput Mbps | 丢包率 | 平均/最大积压包 | p95/p99.9/max ms |", "|---|---:|---:|---:|---:|---:|"]
    for ac in ("vo", "vi"):
        for rate in sorted({float(row["per_sta_target_mbps"]) for row in rows if row["ac"] == ac}):
            subset = group(rows, ac, rate)
            lines.append(f"| {ac.upper()} | {rate:.2f} | {avg(subset,'offered_mbps'):.3f}/{avg(subset,'goodput_mbps'):.3f} | {avg(subset,'drop_rate'):.4%} | {avg(subset,'pending_mean'):.1f}/{avg(subset,'pending_max'):.1f} | {avg(subset,'delay_p95_ms'):.3f}/{avg(subset,'delay_p99_9_ms'):.3f}/{avg(subset,'delay_max_ms'):.3f} |")
    lines += ["", "VO 的排队拐点在每 STA 0.75–1.00 Mbps 之间；VI 在 2.50–3.00 Mbps 之间。当前 S2 的 VO/VI 仅 0.348/0.812 Mbps，S4 为 0.262/0.721 Mbps。", "",
              "## 活跃 AC 队列扩大 10 倍", "",
              "| AC/负载 | 原容量→扩大容量 | 丢包率 | p99.9 ms | 平均积压 | 溢出/过期/重试丢弃 |", "|---|---:|---:|---:|---:|---:|"]
    for ac, rate in (("vo", 1.0), ("vo", 1.5), ("vi", 3.0)):
        base = group(rows, ac, rate); enlarged = group(large, ac, rate)
        original_limit = 100 if ac == "vo" else 200; new_limit = int(avg(enlarged, "queue_limit_packets"))
        fmt = lambda subset, key, percent=False: (f"{avg(subset,key):.3%}" if percent else f"{avg(subset,key):.1f}")
        counts = "/".join(f"{avg(enlarged,key):.1f}" for key in ("overflow_count","expired_count","failed_count"))
        lines.append(f"| {ac.upper()} {rate:.2f} | {original_limit}→{new_limit} | {fmt(base,'drop_rate',True)}→{fmt(enlarged,'drop_rate',True)} | {fmt(base,'delay_p99_9_ms')}→{fmt(enlarged,'delay_p99_9_ms')} | {fmt(base,'pending_mean')}→{fmt(enlarged,'pending_mean')} | {counts} |")
    lines += ["", "- VO 1.00 Mbps/STA 在容量扩大后丢包、积压和尾时延近似不变，证明拐点不是 100 包上限伪造的。",
              "- 在持续过载点扩大 buffer 会显著增加积压和时延：VO 1.50 Mbps/STA 的 p99.9 从约 1.05 秒增至 5.33 秒；VI 3.00 Mbps/STA 从约 161 ms 增至 1.33 秒。",
              "- 大 buffer 能把部分溢出转成等待或成功交付，但没有提高服务能力；这是 bufferbloat。主实验不应靠调大队列来获得预期长尾。", "",
              "## 对混合场景的约束", "",
              "新的实时压力场景应使用每 STA 绝对速率，使 VO 接近 0.8–1.0 Mbps、VI 接近 2.5–3.0 Mbps，再加入受控的 BE/BK。最终点必须用混合流量复测，因为不同 AC 竞争会移动单 AC 拐点。",
              "主队列容量保持 100/200/500/500，寿命 10 秒、retry limit 7；扩大队列只保留为敏感性对照。"]
    report = ROOT / "reports/VO_VI负载标定与队列敏感性_20261019.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report); print(out)


if __name__ == "__main__":
    main()
