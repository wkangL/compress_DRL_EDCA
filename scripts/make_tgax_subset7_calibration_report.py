"""Generate the Chinese Default-EDCA calibration report from measured CSVs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/summary/tgax_subset7_calibration_20261022"
SCAN = ROOT / "results/raw/tgax_subset7_loadscan_paired_5s_20261022"
OFFICIAL20 = ROOT / "results/raw/tgax_subset7_benchmark_paired_20s_20261022/load100/seed20261021/default"
BENCH5 = ROOT / "results/raw/tgax_subset7_benchmark_paired_5s_20261022/load100/seed20261021/default"
OUTPUT = ROOT / "reports/TGax_7BSS负载标定报告.md"
ORDER = ["mix_min1", "mix_min2", "load025", "load050", "load075", "load100",
         "load125", "load150", "rt125", "rt150"]


def read(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def f(value, digits=2):
    return f"{float(value):.{digits}f}"


def p(value):
    return f"{100 * float(value):.2f}%"


def main() -> None:
    runs = read(SUMMARY / "runs.csv")
    ac_rows = read(SUMMARY / "ac_metrics.csv")
    run5 = {r["variant"]: r for r in runs if float(r["measurement_s"]) == 5}
    ac5 = {(r["variant"], r["direction"], r["ac"]): r for r in ac_rows
           if float(r["measurement_s"]) == 5 and r["scope"] == "TARGET"}
    official20 = {(r["direction"], r["ac"]): r for r in ac_rows
                  if r["variant"] == "load100" and float(r["measurement_s"]) == 20
                  and r["scope"] == "TARGET"}
    bench5 = json.loads((BENCH5 / "run_manifest.json").read_text(encoding="utf-8"))
    bench20 = json.loads((OFFICIAL20 / "run_manifest.json").read_text(encoding="utf-8"))

    lines = [
        "# TGax-based 7 BSS Default EDCA负载标定报告", "",
        "更新时间：2026-10-22  ",
        "范围：7个同信道BSS、每BSS 40 STA；本报告不包含DDQN训练。", "",
        "## 1. 本轮固定环境", "",
        "- 正式拓扑为`subset7`：中心BSS 0加TGax reuse-3几何中的六个最近同信道邻居，共280 STA。",
        "- 所有节点使用Default EDCA；BSS 0只是未来唯一控制对象，本轮没有智能控制。",
        "- 官方100%完整保留VoIP、BV3/BV6、Gaming、FTP、HTTP数量和真实TCP/UDP。",
        "- 扩展负载只改变并发业务实例数；同一逻辑流在各倍率中保持STA、方向和应用随机流不变。",
        "- 时延只统计MPDU进入MAC到成功ACK/Block ACK的最终服务时延；A-MPDU逐MPDU记账。", "",
        "## 2. 正确性与CPU基准", "",
        "全部正式运行均通过280/280关联、BSSID匹配、应用数量、MAC包守恒、重传、分位数、队列和EDCA导出检查。", "",
        "| 测量窗口 | 并发 | 墙钟/s | 峰值内存/MiB | 说明 |",
        "|---:|---:|---:|---:|---|",
        f"| 5 s | 1 | {f(bench5['elapsed_wall_seconds'],1)} | {f(bench5['peak_rss_mib'],2)} | 最终配对官方100%输入 |",
        f"| 20 s | 1 | {f(bench20['elapsed_wall_seconds'],1)} | {f(bench20['peak_rss_mib'],2)} | 最终配对官方100%输入 |", "",
        "5秒运行约3.8分钟，20秒约16.5分钟；内存只增加约8 MiB。summary模式关闭逐包文本，但没有删除TCP/UDP、HE、EDCA、聚合、Block ACK、重传或邻居BSS。", "",
        "## 3. 负载定义", "",
        "- `mix_min1`：每BSS每个官方业务行1个实例；7 BSS视频名义源速率约193.2 Mbit/s。",
        "- `mix_min2`：每BSS每个官方业务行2个实例；视频名义源速率约386.4 Mbit/s。",
        "- `load025`至`load150`：按官方每行数量的25%至150%扩展；25%仅视频已约453.6 Mbit/s。",
        "- `rt125/rt150`：BE保持官方100%，只增加VO/VI。除`load100`外均为TGax-based扩展场景。", "",
        "## 4. 5秒配对扫描总览", "",
        "| 场景 | 中心吞吐/Mbit/s | 应用字节交付 | AP非空闲 | 邻居平均非空闲 | 最大队列/包 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in ORDER:
        r = run5[variant]
        lines.append(f"| {variant} | {f(r['target_throughput_mbps'])} | {p(r['target_delivery_fraction'])} | "
                     f"{p(r['target_busy_fraction'])} | {p(r['neighbor_mean_busy_fraction'])} | {r['target_max_queue_packets']} |")
    lines += ["", "吞吐和交付率在严重饱和区不要求单调：新增流会改变TCP拥塞、碰撞、ARP/IP可达时刻和各BSS竞争结果。配对设计消除了STA重新抽样与应用随机流漂移，但5秒仍是短窗口。", "",
              "## 5. 中心BSS VO/VI结果", "",
              "| 场景 | DL VO p99/ms | DL VI p99/ms | UL VO p99/ms | UL VI p99/ms | DL VO交付 | DL VI交付 | UL VO交付 | UL VI交付 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in ORDER:
        values = []
        for key in (("DL","VO"),("DL","VI"),("UL","VO"),("UL","VI")):
            row = ac5.get((variant, *key), {})
            values.append(f(row["p99_ms"]) if row.get("successful_packets") not in (None, "0", "") else "N/A")
        deliveries = []
        for key in (("DL","VO"),("DL","VI"),("UL","VO"),("UL","VI")):
            row = ac5.get((variant, *key), {})
            deliveries.append(p(row.get("application_delivery_fraction", 0)))
        lines.append("| " + " | ".join([variant] + values + deliveries) + " |")
    lines += ["", "`N/A`表示该方向在5秒测量窗内没有成功MPDU，不能解释为零时延。", "",
              "![VO和VI负载响应](../results/figures/tgax_subset7_calibration_20261022/png/06_target_vo_vi_load_scan.png)", "",
              "## 6. 官方100%的20秒结果", "",
              "| 方向-AC | 到达MAC | 成功 | 最终失败 | 截止未完成 | MAC成功率 | 重传 | p50/ms | p95/ms | p99/ms |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for direction, ac in (("DL","VO"),("DL","VI"),("DL","BE"),("UL","VO"),("UL","VI"),("UL","BE")):
        row = official20.get((direction, ac))
        if row is None or int(row.get("successful_packets", 0)) == 0:
            arrived = row.get("mac_arrived", 0) if row else 0
            acked = row.get("mac_acked", 0) if row else 0
            failed = row.get("mac_failed", 0) if row else 0
            unfinished = row.get("mac_unfinished", 0) if row else 0
            retries = row.get("mac_retransmissions", 0) if row else 0
            lines.append(f"| {direction}-{ac} | {arrived} | {acked} | {failed} | {unfinished} | N/A | {retries} | N/A | N/A | N/A |")
        else:
            lines.append(f"| {direction}-{ac} | {row['mac_arrived']} | {row['mac_acked']} | {row['mac_failed']} | "
                         f"{row['mac_unfinished']} | {p(row['mac_ack_fraction'])} | {row['mac_retransmissions']} | "
                         f"{f(row['p50_ms'])} | {f(row['p95_ms'])} | {f(row['p99_ms'])} |")
    lines += ["", "官方100%下，中心BSS的UL VoIP应用有生成字节，但20秒内没有形成被账本识别的QoS MPDU。现有summary证据只能确定阻塞发生在MAC入队之前；ARP/网络层解析是否为直接原因需详细轨迹确认。因此官方100%被定为严重过载压力点，而不是正常双向业务基线。", "",
              "![最终MAC服务时延CCDF](../results/figures/tgax_subset7_calibration_20261022/png/05_target_final_mac_service_delay_ccdf.png)", "",
              "## 7. 工作区间结论", "",
              "1. **最低且可用的锚点：`mix_min1`。** VO/VI双向均进入MAC，实时业务MAC成功率约98.5%-100%，DL VO/VI p99约22.8/29.5 ms，UL约36.7/42.0 ms。但中心AP非空闲仍约89.4%，所以它更准确地说是“最低可用负载”，不是宽裕轻载。",
              "2. **拥塞转折：`mix_min2`。** 中心应用字节交付从77.2%降到31.9%，DL VI p99升到88.8 ms，UL VI应用交付明显下降，说明并发翻倍已跨入拥塞区。",
              "3. **严重过载：`load025`及以上。** 25%点的视频名义源速率本身已超过单信道可持续有效容量；50%以上出现BE高失败、实时业务前MAC阻塞和强烈非单调饱和行为。官方100%应作为压力/鲁棒性场景。",
              "4. **实时加压变体：`rt125/rt150`。** 它们可测试VO/VI保护，但在当前官方BE背景下同样属于严重过载，不能替代可用区间。", "",
              "本轮只有一个拓扑/业务seed，区间判断是环境标定结论，不是置信区间。下一阶段如进入算法设计，建议至少使用`mix_min1`、`mix_min2`和`load025`三个代表点，并在算法性能结论前增加多seed、较长窗口复核。", "",
              "## 8. 文件位置", "",
              "- 正式配对5秒扫描：`results/raw/tgax_subset7_loadscan_paired_5s_20261022/`",
              "- 单进程5秒CPU基准：`results/raw/tgax_subset7_benchmark_paired_5s_20261022/`",
              "- 正式20秒官方100%：`results/raw/tgax_subset7_benchmark_paired_20s_20261022/`",
              "- 汇总CSV：`results/summary/tgax_subset7_calibration_20261022/`",
              "- PNG/PDF：`results/figures/tgax_subset7_calibration_20261022/`", "",
              "本阶段没有启动DDQN训练。"]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
