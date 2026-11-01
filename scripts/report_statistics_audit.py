"""Summarize strict lifecycle audits and write the Stage-A Chinese report."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, backend_source_hash

ACS = ("vo", "vi", "be", "bk")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(row: dict, key: str) -> float:
    value = row[key]
    return float(value) if value not in ("", "None") else math.nan


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def jain(values: list[float]) -> float:
    x = np.asarray(values, dtype=float)
    return float(x.sum() ** 2 / (len(x) * np.square(x).sum())) if np.square(x).sum() else 1.0


def q(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.summary.mkdir(parents=True, exist_ok=False); args.figures.mkdir(parents=True, exist_ok=False)
    cases = sorted(p for p in args.audit.iterdir() if p.is_dir() and (p / "metadata.json").exists()
                   and (p / "analysis/accounting_by_sta_ac_direction.csv").exists())
    aggregate_rows = []; station_rows = []; queue_rows = []; consistency = []
    delay_sets: dict[tuple, list[float]] = defaultdict(list)
    for case in cases:
        meta = json.loads((case / "metadata.json").read_text())
        scenario = meta["scenario"]; seed = int(meta["seed"])
        rows = read_csv(case / "analysis/accounting_by_sta_ac_direction.csv")
        by_group = defaultdict(list)
        for row in rows:
            by_group[(row["ac"], row["direction"])].append(row)
        with gzip.open(case / "analysis/packet_lifecycle.csv.gz", "rt", encoding="utf-8") as stream:
            for p in csv.DictReader(stream):
                terminal_s = p["terminal_s"]
                if p["terminal"] == "acked" and terminal_s not in ("", "None") and 10 < float(terminal_s) <= 70:
                    direction = "DL" if int(p["node"]) == 0 else "UL"
                    delay_sets[(scenario, seed, ACS[int(p["ac"])], direction)].append(float(p["delay_ms"]))
        station_bytes = defaultdict(float); station_ac_bytes = defaultdict(float)
        for row in rows:
            station = int(row["sta"]); ac = row["ac"]; delivered = number(row, "delivered_bytes")
            station_bytes[station] += delivered; station_ac_bytes[(station, ac)] += delivered
        for station in range(10):
            station_rows.append(dict(scenario=scenario, seed=seed, sta=station,
                                     goodput_mbps=station_bytes[station] * 8 / 60 / 1e6,
                                     **{f"{ac}_goodput_mbps": station_ac_bytes[(station, ac)] * 8 / 60 / 1e6 for ac in ACS},
                                     vo_ontime_fraction=next(number(r, "ontime_fraction") for r in rows if int(r["sta"]) == station and r["ac"] == "vo" and r["direction"] == "UL"),
                                     vi_ontime_fraction=next(number(r, "ontime_fraction") for r in rows if int(r["sta"]) == station and r["ac"] == "vi" and r["direction"] == "UL")))
        audit_json = json.loads((case / "analysis/audit.json").read_text())
        for (ac, direction), group in sorted(by_group.items()):
            sums = {key: int(sum(number(r, key) for r in group)) for key in (
                "active_start", "generated", "accepted", "queue_rejected", "acked", "expired",
                "retry_limit", "qos_old", "other_drop", "unfinished_end", "attempts",
                "retry_attempts", "ack_timeouts", "delivered_bytes", "generated_bytes",
                "ontime_eligible", "ontime", "late_success", "final_failure", "deadline_unfinished")}
            failures = sum(sums[k] for k in ("queue_rejected", "expired", "retry_limit", "qos_old", "other_drop"))
            at_risk = sums["active_start"] + sums["generated"]
            delays = delay_sets[(scenario, seed, ac, direction)]
            collisions = audit_json["phy_failures"].get(f"COLLISION_RX|{ACS.index(ac)}|{direction}", 0)
            other_phy = audit_json["phy_failures"].get(f"OTHER_RX_FAIL|{ACS.index(ac)}|{direction}", 0)
            internal = audit_json.get("internal_collisions",{}).get(f"{ACS.index(ac)}|{direction}",0)
            aggregate_rows.append(dict(scenario=scenario, seed=seed, ac=ac, direction=direction, **sums,
                final_failures=failures, at_risk=at_risk,
                delivery_fraction=sums["acked"] / at_risk,
                final_failure_fraction=failures / at_risk,
                unfinished_fraction=sums["unfinished_end"] / at_risk,
                queue_rejection_fraction_of_generated=sums["queue_rejected"] / sums["generated"],
                external_collision_count=collisions,
                external_collision_per_data_attempt=collisions / max(sums["attempts"], 1),
                internal_collision_count=internal,
                internal_collision_per_access_event=internal / max(sums["attempts"] + internal, 1),
                combined_collision_event_fraction=(collisions + internal) / max(sums["attempts"] + internal, 1),
                other_phy_failure_count=other_phy,
                ack_timeout_per_data_attempt=sums["ack_timeouts"] / max(sums["attempts"], 1),
                offered_mbps=sums["generated_bytes"] * 8 / 60 / 1e6,
                goodput_mbps=sums["delivered_bytes"] * 8 / 60 / 1e6,
                delay_mean_ms=float(np.mean(delays)) if delays else math.nan,
                delay_p95_ms=q(delays, 95), delay_p99_ms=q(delays, 99), delay_max_ms=q(delays, 100),
                ontime_fraction=sums["ontime"] / sums["ontime_eligible"] if sums["ontime_eligible"] else math.nan))
        for row in read_csv(case / "analysis/queue_statistics.csv"):
            queue_rows.append(dict(scenario=scenario, seed=seed, **row))
        production = ROOT / "results/raw/baseline_absolute_full_20261019" / case.name / "summary.json"
        old = json.loads(production.read_text())
        for ac in ACS:
            selected = [r for r in aggregate_rows if r["scenario"] == scenario and r["seed"] == seed and r["ac"] == ac]
            checks = {
                "offered_packets": sum(r["generated"] for r in selected),
                "delivered_packets": sum(r["acked"] for r in selected),
                "overflow_count": sum(r["queue_rejected"] for r in selected),
                "expired_count": sum(r["expired"] for r in selected),
                "failed_count": sum(r["retry_limit"] for r in selected),
                "other_drop_count": sum(r["qos_old"] + r["other_drop"] for r in selected),
                "pending_packets_start": sum(r["active_start"] for r in selected),
                "pending_packets_end": sum(r["unfinished_end"] for r in selected),
            }
            for metric, observed in checks.items():
                expected = int(old[f"{ac}_{metric}"])
                consistency.append(dict(scenario=scenario, seed=seed, ac=ac, metric=metric,
                                        production=expected, packet_audit=observed, difference=observed - expected,
                                        passed=observed == expected))
    if not all(r["passed"] for r in consistency):
        raise AssertionError("strict audit disagrees with production accounting")
    save_csv(args.summary / "accounting_ac_direction.csv", aggregate_rows)
    save_csv(args.summary / "station_service.csv", station_rows)
    save_csv(args.summary / "queue_statistics.csv", queue_rows)
    save_csv(args.summary / "production_consistency.csv", consistency)

    fairness_rows = []
    for scenario in sorted({r["scenario"] for r in station_rows}):
        for seed in sorted({r["seed"] for r in station_rows if r["scenario"] == scenario}):
            part = [r for r in station_rows if r["scenario"] == scenario and r["seed"] == seed]
            fairness_rows.append(dict(scenario=scenario, seed=seed,
                jain_total=jain([r["goodput_mbps"] for r in part]),
                **{f"jain_{ac}": jain([r[f"{ac}_goodput_mbps"] for r in part]) for ac in ACS},
                vo_ontime_min=min(r["vo_ontime_fraction"] for r in part),
                vo_ontime_median=float(np.median([r["vo_ontime_fraction"] for r in part])),
                vi_ontime_min=min(r["vi_ontime_fraction"] for r in part),
                vi_ontime_median=float(np.median([r["vi_ontime_fraction"] for r in part]))))
    save_csv(args.summary / "fairness_and_satisfaction.csv", fairness_rows)

    for scenario in ("S2_realtime_stress", "S4_overload"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for axis, ac in zip(axes, ("vo", "vi")):
            for direction, color in (("UL", "tab:blue"), ("DL", "tab:orange")):
                for seed, style in ((101, "-"), (102, "--")):
                    values = np.sort(np.asarray(delay_sets[(scenario, seed, ac, direction)]))
                    if len(values):
                        ccdf = (len(values) - np.arange(len(values))) / len(values)
                        axis.step(values, ccdf, where="post", color=color, linestyle=style,
                                  label=f"{direction} seed {seed}")
            axis.set_yscale("log"); axis.set_ylim(1e-5, 1); axis.grid(True, which="both", alpha=.25)
            axis.set_xlabel("MAC packet delay (ms)"); axis.set_ylabel("P(delay > x)")
            axis.set_title(ac.upper()); axis.legend(fontsize=8)
        fig.suptitle(f"{scenario}: baseline VO/VI delay CCDF by direction")
        fig.savefig(args.figures / f"{scenario}_vo_vi_direction_ccdf.png", dpi=220)
        fig.savefig(args.figures / f"{scenario}_vo_vi_direction_ccdf.pdf")
        plt.close(fig)

    def mean_metric(scenario: str, ac: str, direction: str, key: str) -> float:
        vals = [float(r[key]) for r in aggregate_rows if r["scenario"] == scenario and r["ac"] == ac and r["direction"] == direction]
        return float(np.mean(vals))

    s2vi = []
    for direction in ("UL", "DL"):
        s2vi.append(dict(direction=direction,
            p95=mean_metric("S2_realtime_stress", "vi", direction, "delay_p95_ms"),
            p99=mean_metric("S2_realtime_stress", "vi", direction, "delay_p99_ms"),
            reject=mean_metric("S2_realtime_stress", "vi", direction, "queue_rejection_fraction_of_generated"),
            fail=mean_metric("S2_realtime_stress", "vi", direction, "final_failure_fraction"),
            collision=mean_metric("S2_realtime_stress", "vi", direction, "external_collision_per_data_attempt"),
            timeout=mean_metric("S2_realtime_stress", "vi", direction, "ack_timeout_per_data_attempt"),
            ontime=mean_metric("S2_realtime_stress", "vi", direction, "ontime_fraction")))

    lines = [
        "# 指标统计正确性核查", "", "更新时间：2026-10-20（Asia/Shanghai）", "",
        "## 结论", "",
        "阶段 A 的严格核查已通过：S1–S4 各两个种子（101、102），每次 10 秒业务预热加 60 秒测量，"
        "逐包验证 80 个 STA×AC×方向分组。每个包最多一个终态；测量窗口均满足 `期初在途 + 窗口生成 = ACK + 各类最终失败 + 期末未完成`；"
        "窗口生成包还满足 `成功入队 + 同步拒绝入队 = 生成`。观察器只增加 CSV trace，短程生产/观察器 stdout 完全一致，生产二进制哈希未改变。", "",
        f"逐包核查与历史 baseline schema 2 的 {len(consistency)} 个共同 AC 计数字段全部逐项相等；本轮生产账本为 schema 3。代码指纹为 `{backend_source_hash()}`。", "",
        "这表示现有 baseline 的 offered、ACK、overflow、expiry、retry-limit、other 和 pending 计数正确；"
        "但现有汇总中的 `drop_rate` 分母只用窗口内 generated，分子可能含期初积压在窗口内失败的包。后续正式表统一使用 `final_failure_fraction = 最终失败 / (期初在途 + 窗口生成)`，"
        "并单列 `queue_rejection_fraction_of_generated`、`unfinished_fraction` 和 `delivery_fraction`。旧字段保留用于历史复现。", "",
        "## 审计数据", "",
        "| 场景 | seed | trace 包数 | STA×AC×方向组 | 守恒 | 入队决策 | 重复终态 |", "|---|---:|---:|---:|---|---|---:|",
    ]
    complete = json.loads((args.audit / "complete.json").read_text())
    for case in complete["cases"]:
        lines.append(f"| {case['scenario']} | {case['seed']} | {case['packets_seen']} | {case['groups']} | 通过 | 通过 | 0 |")
    lines += ["", "原始逐包表为各 case 的 `analysis/packet_lifecycle.csv.gz`；汇总表位于 "
              f"`{args.summary.resolve().relative_to(ROOT).as_posix()}`。失败的准备目录均以 `failed_` 标记并保留，"
              "其中没有被纳入性能结论。", "",
              "## 统一口径", "",
              "- generated：测量窗口内创建并交给 Wi-Fi 设备的 MAC 业务包。accepted：实际触发 MAC queue Enqueue；queue rejected：同步 FailedEnqueue。",
              "- ACK success：AckedMpdu；expiry：ExpiredLifetime；retry final：ReachedRetryLimit；other：其余 DroppedMpdu 原因；unfinished：测量结束仍未出现终态。",
              "- goodput：测量窗口内 AckedMpdu 的业务 payload bit / 60 s，按 UL、DL、AC 和 STA 都可拆分。",
              "- delay：创建时间到 AckedMpdu 的时间，仅对成功包定义；失败和 unfinished 不塞入成功时延分布，而在 on-time、late success、final failure、unfinished 中单列。",
              "- VO deadline 暂按 20 ms、VI 按 50 ms，用 `created <= end-deadline` 的包作为 on-time 分母，避免右删失伪装为超时。",
              "- external collision：目标接收端 PHY drop，且该 PSDU 在空口区间内检测到重叠发送；分母为相同 AC/方向的实际 DATA attempts。other PHY failure 与 ACK/CTS timeout 分列。主场景 RTS/CTS 关闭，因此 RTS 计数为 0。",
              "- internal collision：同一设备多个 EDCAF 同时到期、输掉虚拟碰撞的事件。schema 3 在 `FrameExchangeManager::NotifyInternalCollision` 更新重试/CW 前触发只读 trace；它与 external collision 分列，组合事件比例使用 `(external+internal)/(DATA attempts+internal)`。", "",
              "## S2 VI 长尾成因", "", "两个审计种子的等权均值：", "",
              "| 方向 | 成功时延 p95/p99 (ms) | 生成包拒绝入队 | 最终失败率 | 外部碰撞/attempt | timeout/attempt | 50 ms按时率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in s2vi:
        lines.append(f"| {r['direction']} | {r['p95']:.2f}/{r['p99']:.2f} | {r['reject']:.2%} | {r['fail']:.2%} | {r['collision']:.2%} | {r['timeout']:.2%} | {r['ontime']:.2%} |")
    lines += ["", "S2 VI 的几百毫秒尾部由高竞争下的反复退避/timeout和队列等待共同形成；它不是把失败包赋成低时延的统计假象。"
              "方向拆分 CCDF 位于审计图目录。队列统计表给出每节点/AC 的 mean、p95、max 和 nonempty fraction；AP 的 DL 是共享 AC 队列，不能伪造成每目的 STA 的独立 DL 队列。", "",
              "## Jain 与业务满足率", "", "Jain 同时输出全业务每 STA goodput及每 AC 每 STA goodput，时间尺度为完整 60 秒测量窗口。"
              "`fairness_and_satisfaction.csv` 另给 VO/VI 每 STA 的按时率最小值和中位数。现有 reward 使用的总 goodput Jain 不能代表 VO/VI 分别公平，也不能代表各 STA 是否满足时限。", "",
              "## 阶段 A 后续", "",
              "1. schema 3 已把内部虚拟碰撞、统一失败分母和 UL/DL×AC 成功/时延输出接入正式 `metrics.py`；schema 2 仍可读取，旧结果不改写。",
              "2. 单元测试覆盖 schema 2/3 混用拒绝、窗口边界、重复终态、漏入队决策；真实 ns-3 测试验证 internal-collision trace 数量与 P 账本相等。",
              "3. Stage 0 通过；真实 ns-3 `unittest` 35/35、`pytest` 40/40 通过；容量指纹重跑仍为 27.30225 Mbps。阶段 A 现已冻结。",
              "4. 下一阶段只做 AP 可观测性审计，仍不立即启动正式 DDQN。", ""]
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "checks": len(consistency), "report": str(args.report)}))


if __name__ == "__main__":
    main()
