"""Analyze the three-seed single-BSS direct-MAC full-buffer Default baseline."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT


RUN = ROOT / "results/raw/mac_fullbuffer40_default_20s_3seed_20261028"
INPUT = ROOT / "results/raw/tgax_mac_fullbuffer40_inputs_20261028"
OUT = ROOT / "results/summary/mac_fullbuffer40_default_20s_3seed_20261028"
FIG = ROOT / "results/figures/mac_fullbuffer40_default_20s_3seed_20261028"
ACS = ("VO", "VI", "BE", "BK")
COLORS = {"VO": "#d62728", "VI": "#ff8c00", "BE": "#3973b7", "BK": "#7f7f7f"}
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False})


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def ratio(a: float, b: float) -> float | None:
    return a / b if b else None


def jain(values: list[float]) -> float | None:
    denominator = len(values) * sum(value * value for value in values)
    return sum(values) ** 2 / denominator if denominator else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    sources = read(INPUT / "fullbuffer_sources.csv")
    topology = read(INPUT / "topology.csv")
    seeds = (41, 42, 43)
    per_seed: list[dict] = []
    validation: dict[str, dict] = {}

    for seed in seeds:
        run = RUN / f"seed_{seed}"
        outcomes = [row for row in read(run / "mac_qos_outcomes.csv")
                    if row["scope"] == "BSS" and row["direction"] == "UL"]
        delays = {row["ac"]: row for row in read(run / "mac_service_delay_summary.csv")
                  if row["scope"] == "BSS" and row["direction"] == "UL"}
        source_outcomes = read(run / "fullbuffer_source_outcomes.csv")
        queue_stats = read(run / "fullbuffer_queue_stats.csv")
        competition = read(run / "competition_samples.csv")
        bss = read(run / "bss_summary.csv")[0]
        audit = read(run / "fullbuffer_audit.csv")[0]
        associations = read(run / "associations.csv")
        ampdu = {row["ac"]: row for row in read(run / "ampdu_summary.csv")}
        ip_rows = read(run / "ip_flow_summary.csv")

        checks = {
            "40_of_40_associated": len(associations) == 40 and all(row["associated"] == "1" for row in associations),
            "all_bssid_match": all(row["match"] == "1" for row in associations),
            "exact_72_sources": len(source_outcomes) == 72 and int(audit["active_sources"]) == 72,
            "no_ipv4_stack": audit["ipv4_stack_installed"] == "0",
            "no_ip_flows": len(ip_rows) == 0,
            "no_duplicate_admission": audit["duplicate_mac_admissions"] == "0",
            "four_ul_acs": {row["ac"] for row in outcomes} == set(ACS),
            "ledger_conserved": all(int(row["arrived"]) == int(row["acked"]) +
                                    int(row["failed"]) + int(row["unfinished"]) for row in outcomes),
            "all_source_queues_nonempty_95pct": len(queue_stats) == 72 and
                                                all(float(row["nonempty_fraction"]) >= .95 for row in queue_stats),
            "vo_vi_ampdu_observed": all(int(ampdu[ac]["max_mpdus_per_psdu"]) > 1 for ac in ("VO", "VI")),
        }
        validation[str(seed)] = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}

        row: dict = {
            "seed": seed,
            "total_throughput_mbps": float(bss["throughput_mbps"]),
            "busy_fraction": float(bss["busy_fraction"]),
            "mean_active_stas": float(np.mean([float(item["active_stas"]) for item in competition])),
            "min_active_stas": min(int(item["active_stas"]) for item in competition),
            "mean_active_queues": float(np.mean([float(item["active_source_queues"]) for item in competition])),
            "min_queue_nonempty_fraction": min(float(item["nonempty_fraction"]) for item in queue_stats),
        }
        for ac in ACS:
            outcome = next(item for item in outcomes if item["ac"] == ac)
            ac_sources = [item for item in source_outcomes if item["ac"] == ac]
            ac_queues = [item for item in queue_stats if item["ac"] == ac]
            delivered = [int(item["acked_payload_bytes"]) * 8 / 20 / 1e6 for item in ac_sources]
            row.update({
                f"{ac}_throughput_mbps": int(outcome["acked_bytes"]) * 8 / 20 / 1e6,
                f"{ac}_p95_ms": float(delays[ac]["p95_ms"]) if int(delays[ac]["successful_packets"]) else None,
                f"{ac}_p99_ms": float(delays[ac]["p99_ms"]) if int(delays[ac]["successful_packets"]) else None,
                f"{ac}_successful_mpdus": int(outcome["acked"]),
                f"{ac}_failure_rate": ratio(int(outcome["failed"]), int(outcome["arrived"])),
                f"{ac}_expiry_rate": ratio(int(outcome["expired_lifetime"]), int(outcome["arrived"])),
                f"{ac}_unfinished_rate": ratio(int(outcome["unfinished"]), int(outcome["arrived"])),
                f"{ac}_on_time_rate": ratio(int(outcome["on_time_delivered"]), int(outcome["deadline_eligible"])),
                f"{ac}_jain": jain(delivered),
                f"{ac}_source_count": len(ac_sources),
                f"{ac}_mean_queue_packets": float(np.mean([float(item["mean_packets"]) for item in ac_queues])),
                f"{ac}_queue_nonempty_fraction": float(np.mean([float(item["nonempty_fraction"]) for item in ac_queues])),
                f"{ac}_ampdu_fraction": float(ampdu[ac]["ampdu_fraction"]),
                f"{ac}_mean_mpdus_per_psdu": float(ampdu[ac]["mean_mpdus_per_psdu"]),
                f"{ac}_max_mpdus_per_psdu": int(ampdu[ac]["max_mpdus_per_psdu"]),
            })
        per_seed.append(row)

    write(OUT / "per_seed_metrics.csv", per_seed)
    means: list[dict] = []
    for ac in ACS:
        row = {"ac": ac}
        for suffix in ("throughput_mbps", "p95_ms", "p99_ms", "failure_rate", "expiry_rate",
                       "unfinished_rate", "on_time_rate", "jain", "mean_queue_packets",
                       "queue_nonempty_fraction", "ampdu_fraction", "mean_mpdus_per_psdu"):
            values = [item[f"{ac}_{suffix}"] for item in per_seed if item[f"{ac}_{suffix}"] is not None]
            row[f"{suffix}_mean"] = float(np.mean(values)) if values else None
            row[f"{suffix}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        means.append(row)
    write(OUT / "ac_mean_std.csv", means)
    (OUT / "validation.json").write_text(json.dumps({
        "status": "PASS" if all(item["status"] == "PASS" for item in validation.values()) else "FAIL",
        "seeds": validation,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # Topology and exact source coverage.
    source_map = defaultdict(set)
    for item in sources:
        source_map[item["source"]].add(item["ac"])
    fig, ax = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    ap = next(item for item in topology if item["role"] == "AP")
    ax.scatter([float(ap["x"])], [float(ap["y"])], marker="*", s=320, c="#111111", label="AP")
    for primary in ("VO", "VI"):
        items = [item for item in topology if item["role"] == "STA" and primary in source_map[item["node_label"]]]
        ax.scatter([float(item["x"]) for item in items], [float(item["y"]) for item in items],
                   s=64, c=COLORS[primary], label=f"{primary}主业务（20 STA）", alpha=.85)
    for ac, marker, size in (("BE", "o", 125), ("BK", "s", 115)):
        items = [item for item in topology if item["role"] == "STA" and ac in source_map[item["node_label"]]]
        ax.scatter([float(item["x"]) for item in items], [float(item["y"]) for item in items],
                   s=size, facecolors="none", edgecolors=COLORS[ac], marker=marker,
                   linewidths=1.4, label=f"附加{ac}（{len(items)} STA）")
    ax.set(xlabel="x / m", ylabel="y / m", title="单BSS 40 STA真实位置与Full Buffer业务覆盖")
    ax.set_aspect("equal"); ax.grid(alpha=.2); ax.legend(fontsize=9, ncol=2)
    fig.savefig(FIG / "topology_and_source_coverage.png", dpi=220); plt.close(fig)

    # Throughput.
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    x = np.arange(4); values = [next(item for item in means if item["ac"] == ac)["throughput_mbps_mean"] for ac in ACS]
    errors = [next(item for item in means if item["ac"] == ac)["throughput_mbps_std"] for ac in ACS]
    ax.bar(x, values, yerr=errors, capsize=5, color=[COLORS[ac] for ac in ACS], alpha=.82)
    for index, ac in enumerate(ACS):
        ax.scatter(np.full(3, index), [row[f"{ac}_throughput_mbps"] for row in per_seed], c="black", s=20, zorder=3)
    ax.set(xticks=x, xticklabels=ACS, ylabel="成功MAC有效载荷吞吐量 (Mbps)",
           title="Default EDCA：各AC饱和上行吞吐（三seed均值±样本标准差）")
    ax.grid(axis="y", alpha=.22); fig.savefig(FIG / "ac_throughput.png", dpi=220); plt.close(fig)

    # VO/VI service delay CCDF per seed.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, ac in zip(axes, ("VO", "VI")):
        for seed in seeds:
            rows = [item for item in read(RUN / f"seed_{seed}/mac_service_delay_ccdf.csv")
                    if item["scope"] == "BSS" and item["direction"] == "UL" and item["ac"] == ac]
            ax.plot([float(item["delay_ms"]) for item in rows],
                    [float(item["ccdf"]) for item in rows], label=f"seed {seed}", linewidth=1.3)
        ax.set_yscale("log"); ax.set(xlabel="MAC最终服务时延 (ms)", ylabel="P(delay > x)", title=f"{ac} CCDF")
        ax.grid(True, which="both", alpha=.22); ax.legend()
    fig.savefig(FIG / "vo_vi_service_delay_ccdf.png", dpi=220); plt.close(fig)

    # Failure and fairness.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    width = .36
    failures = [next(item for item in means if item["ac"] == ac)["failure_rate_mean"] for ac in ACS]
    expiries = [next(item for item in means if item["ac"] == ac)["expiry_rate_mean"] for ac in ACS]
    axes[0].bar(x-width/2, failures, width, label="最终失败率", color="#c44e52")
    axes[0].bar(x+width/2, expiries, width, label="其中队列过期率", color="#dd8452")
    axes[0].set(xticks=x, xticklabels=ACS, ylabel="比例", title="最终失败与队列过期")
    axes[0].legend(); axes[0].grid(axis="y", alpha=.22)
    fairness = [next(item for item in means if item["ac"] == ac)["jain_mean"] for ac in ACS]
    axes[1].bar(x, fairness, color=[COLORS[ac] for ac in ACS])
    axes[1].set(xticks=x, xticklabels=ACS, ylim=(0, 1.03), ylabel="STA吞吐Jain", title="各AC内部STA公平性")
    axes[1].grid(axis="y", alpha=.22)
    fig.savefig(FIG / "failure_expiry_and_jain.png", dpi=220); plt.close(fig)

    # Queue and contention diagnostics.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    queue_nonempty = [next(item for item in means if item["ac"] == ac)["queue_nonempty_fraction_mean"] for ac in ACS]
    queue_depth = [next(item for item in means if item["ac"] == ac)["mean_queue_packets_mean"] for ac in ACS]
    axes[0].bar(x, queue_nonempty, color=[COLORS[ac] for ac in ACS]); axes[0].set(xticks=x, xticklabels=ACS, ylim=(0,1.02), title="队列非空时间比例")
    axes[1].bar(x, queue_depth, color=[COLORS[ac] for ac in ACS]); axes[1].set(xticks=x, xticklabels=ACS, title="活跃源平均队列深度", ylabel="packet")
    axes[2].bar([str(seed) for seed in seeds], [row["mean_active_stas"] for row in per_seed], color="#4c72b0")
    axes[2].set(ylim=(0, 41), xlabel="seed", ylabel="STA数", title="同时有上行待发数据的STA")
    for ax in axes: ax.grid(axis="y", alpha=.22)
    fig.savefig(FIG / "queue_and_contention_diagnostics.png", dpi=220); plt.close(fig)

    total = [row["total_throughput_mbps"] for row in per_seed]
    busy = [row["busy_fraction"] for row in per_seed]
    summary = {
        "validation": "PASS" if all(item["status"] == "PASS" for item in validation.values()) else "FAIL",
        "seeds": list(seeds), "duration_s": 20,
        "total_throughput_mbps_mean": float(np.mean(total)),
        "total_throughput_mbps_std": float(np.std(total, ddof=1)),
        "busy_fraction_mean": float(np.mean(busy)),
        "mean_active_stas": float(np.mean([row["mean_active_stas"] for row in per_seed])),
        "minimum_active_stas_across_samples": min(row["min_active_stas"] for row in per_seed),
        "minimum_queue_nonempty_fraction": min(row["min_queue_nonempty_fraction"] for row in per_seed),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
