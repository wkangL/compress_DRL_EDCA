"""Analyze the three-seed one-BSS direct-MAC TGax-arrival baseline."""
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


RUN = ROOT / "results/raw/mac_arrival40_default_20s_3seed_20261029"
INPUT = ROOT / "results/raw/tgax_mac_arrival40_inputs_20261029"
OUT = ROOT / "results/summary/mac_arrival40_default_20s_3seed_20261029"
FIG = ROOT / "results/figures/mac_arrival40_default_20s_3seed_20261029"
FULL = ROOT / "results/summary/mac_fullbuffer40_default_20s_3seed_20261028"
ACS = ("VO", "VI", "BE", "BK")
DIRECTIONS = ("UL", "DL")
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
    seeds = (41, 42, 43)
    per_seed: list[dict] = []
    model_rows: list[dict] = []
    validation: dict[str, dict] = {}

    for seed in seeds:
        run = RUN / f"seed_{seed}"
        outcomes = [row for row in read(run / "mac_qos_outcomes.csv") if row["scope"] == "BSS"]
        delays = {(row["direction"], row["ac"]): row
                  for row in read(run / "mac_service_delay_summary.csv") if row["scope"] == "BSS"}
        flows = read(run / "direct_arrival_flows.csv")
        queues = read(run / "fullbuffer_queue_stats.csv")
        sources = read(run / "fullbuffer_source_outcomes.csv")
        competition = read(run / "competition_samples.csv")
        audit = read(run / "direct_arrival_audit.csv")[0]
        associations = read(run / "associations.csv")
        bss = read(run / "bss_summary.csv")[0]
        models = {row["model"] for row in flows if int(row["generated_events"]) > 0}
        checks = {
            "40_of_40_associated": len(associations) == 40 and all(row["associated"] == "1" for row in associations),
            "all_bssid_match": all(row["match"] == "1" for row in associations),
            "no_ipv4_stack": audit["ipv4_stack_installed"] == "0",
            "no_ip_flows": len(read(run / "ip_flow_summary.csv")) == 0,
            "all_runtime_transports_none": all(row["transport"] == "NONE" for row in flows),
            "all_models_generated": {"VOIP", "GMG", "BV3", "FTP", "HTTP", "HTTP_REQ"}.issubset(models),
            "direct_demand_conserved": audit["byte_conservation"] == "1",
            "no_duplicate_admission": audit["duplicate_mac_admissions"] == "0",
            "four_acs_both_directions": {(row["direction"], row["ac"]) for row in outcomes} ==
                                          {(direction, ac) for direction in DIRECTIONS for ac in ACS},
            "mac_ledger_conserved": all(int(row["arrived"]) == int(row["acked"]) +
                                         int(row["failed"]) + int(row["unfinished"]) for row in outcomes),
        }
        validation[str(seed)] = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}

        row: dict = {
            "seed": seed,
            "total_throughput_mbps": float(bss["throughput_mbps"]),
            "busy_fraction": float(bss["busy_fraction"]),
            "generated_bytes": int(audit["generated_bytes"]),
            "injected_bytes": int(audit["injected_bytes"]),
            "pending_app_bytes": int(audit["pending_bytes"]),
            "mean_active_stas": float(np.mean([float(item["active_stas"]) for item in competition])),
            "max_active_stas": max(int(item["active_stas"]) for item in competition),
            "mean_active_queues": float(np.mean([float(item["active_source_queues"]) for item in competition])),
        }
        for direction in DIRECTIONS:
            for ac in ACS:
                outcome = next(item for item in outcomes if item["direction"] == direction and item["ac"] == ac)
                delay = delays[(direction, ac)]
                key = f"{direction}_{ac}"
                row.update({
                    f"{key}_throughput_mbps": int(outcome["acked_bytes"]) * 8 / 20 / 1e6,
                    f"{key}_successful_mpdus": int(outcome["acked"]),
                    f"{key}_p95_ms": float(delay["p95_ms"]) if int(delay["successful_packets"]) else None,
                    f"{key}_p99_ms": float(delay["p99_ms"]) if int(delay["successful_packets"]) else None,
                    f"{key}_failure_rate": ratio(int(outcome["failed"]), int(outcome["arrived"])),
                    f"{key}_expiry_rate": ratio(int(outcome["expired_lifetime"]), int(outcome["arrived"])),
                    f"{key}_unfinished_rate": ratio(int(outcome["unfinished"]), int(outcome["arrived"])),
                    f"{key}_on_time_rate": ratio(int(outcome["on_time_delivered"]), int(outcome["deadline_eligible"])),
                })
        for ac in ACS:
            ac_sources = [item for item in sources if item["ac"] == ac and item["source"] != "ap0"]
            row[f"UL_{ac}_jain"] = jain([int(item["acked_payload_bytes"]) * 8 / 20 / 1e6 for item in ac_sources])
            ac_queues = [item for item in queues if item["ac"] == ac]
            row[f"{ac}_queue_nonempty_fraction"] = float(np.mean([float(item["nonempty_fraction"]) for item in ac_queues])) if ac_queues else None
        per_seed.append(row)

        grouped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for flow in flows:
            model = flow["model"]
            for field in ("generated_events", "generated_bytes", "injected_packets", "injected_bytes", "pending_bytes"):
                grouped[model][field] += int(flow[field])
        for model, values in sorted(grouped.items()):
            model_rows.append({"seed": seed, "model": model, **values})

    write(OUT / "per_seed_metrics.csv", per_seed)
    write(OUT / "per_seed_model_demand.csv", model_rows)
    (OUT / "validation.json").write_text(json.dumps({
        "status": "PASS" if all(item["status"] == "PASS" for item in validation.values()) else "FAIL",
        "seeds": validation,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    aggregates: list[dict] = []
    for direction in DIRECTIONS:
        for ac in ACS:
            item: dict = {"direction": direction, "ac": ac}
            for suffix in ("throughput_mbps", "successful_mpdus", "p95_ms", "p99_ms",
                           "failure_rate", "expiry_rate", "unfinished_rate", "on_time_rate"):
                values = [row[f"{direction}_{ac}_{suffix}"] for row in per_seed
                          if row[f"{direction}_{ac}_{suffix}"] is not None]
                item[f"{suffix}_mean"] = float(np.mean(values)) if values else None
                item[f"{suffix}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                item[f"{suffix}_defined_seeds"] = len(values)
            fairness = [row[f"UL_{ac}_jain"] for row in per_seed if row[f"UL_{ac}_jain"] is not None] if direction == "UL" else []
            item["jain_mean"] = float(np.mean(fairness)) if fairness else None
            item["jain_std"] = float(np.std(fairness, ddof=1)) if len(fairness) > 1 else 0.0
            aggregates.append(item)
    write(OUT / "direction_ac_mean_std.csv", aggregates)

    # Traffic demand and MAC admission by model.
    models = ("VOIP", "GMG", "BV3", "FTP", "HTTP", "HTTP_REQ")
    fig, ax = plt.subplots(figsize=(9.4, 5.4), constrained_layout=True)
    x = np.arange(len(models)); width = .27
    generated = [np.mean([r["generated_bytes"] for r in model_rows if r["model"] == model]) / 1e6 for model in models]
    injected = [np.mean([r["injected_bytes"] for r in model_rows if r["model"] == model]) / 1e6 for model in models]
    pending = [np.mean([r["pending_bytes"] for r in model_rows if r["model"] == model]) / 1e6 for model in models]
    ax.bar(x-width, generated, width, label="生成需求", color="#4c72b0")
    ax.bar(x, injected, width, label="提交MAC", color="#55a868")
    ax.bar(x+width, pending, width, label="应用侧待提交", color="#c44e52")
    ax.set(xticks=x, xticklabels=models, ylabel="1秒业务预热+20秒测量的字节量（三seed均值，MB）",
           title="TGax到达模型：需求生成与MAC提交")
    ax.legend(); ax.grid(axis="y", alpha=.22)
    fig.savefig(FIG / "model_demand_and_mac_admission.png", dpi=220); plt.close(fig)

    # Per-direction throughput.
    fig, ax = plt.subplots(figsize=(9, 5.3), constrained_layout=True)
    x = np.arange(4); width = .34
    for offset, direction, color in ((-width/2, "UL", "#4c72b0"), (width/2, "DL", "#dd8452")):
        values = [np.mean([r[f"{direction}_{ac}_throughput_mbps"] for r in per_seed]) for ac in ACS]
        errors = [np.std([r[f"{direction}_{ac}_throughput_mbps"] for r in per_seed], ddof=1) for ac in ACS]
        ax.bar(x+offset, values, width, yerr=errors, capsize=4, label=direction, color=color, alpha=.86)
    ax.set(xticks=x, xticklabels=ACS, ylabel="成功MAC有效载荷吞吐量 (Mbps)",
           title="随机到达Default EDCA：各AC上下行吞吐")
    ax.legend(); ax.grid(axis="y", alpha=.22)
    fig.savefig(FIG / "direction_ac_throughput.png", dpi=220); plt.close(fig)

    # VO/VI CCDF, directions kept distinct.
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), constrained_layout=True)
    for ax, (ac, direction) in zip(axes.flat, (("VO", "UL"), ("VO", "DL"), ("VI", "UL"), ("VI", "DL"))):
        for seed in seeds:
            rows = [item for item in read(RUN / f"seed_{seed}/mac_service_delay_ccdf.csv")
                    if item["scope"] == "BSS" and item["direction"] == direction and item["ac"] == ac]
            if not rows:
                continue
            ax.plot([float(item["delay_ms"]) for item in rows], [float(item["ccdf"]) for item in rows],
                    label=f"seed {seed}", linewidth=1.25)
        ax.set_yscale("log"); ax.set(xlabel="MAC最终服务时延 (ms)", ylabel="P(delay > x)", title=f"{ac}-{direction}")
        ax.grid(True, which="both", alpha=.22)
        handles, labels_here = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    fig.savefig(FIG / "vo_vi_service_delay_ccdf.png", dpi=220); plt.close(fig)

    # Failure/expiry and timely delivery.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    labels = [f"{ac}-{direction}" for ac in ACS for direction in DIRECTIONS]
    failure = [np.mean([r[f"{direction}_{ac}_failure_rate"] for r in per_seed]) for ac in ACS for direction in DIRECTIONS]
    expiry = [np.mean([r[f"{direction}_{ac}_expiry_rate"] for r in per_seed]) for ac in ACS for direction in DIRECTIONS]
    xx = np.arange(len(labels)); w = .38
    axes[0].bar(xx-w/2, failure, w, label="最终失败", color="#c44e52")
    axes[0].bar(xx+w/2, expiry, w, label="队列过期", color="#dd8452")
    axes[0].set(xticks=xx, xticklabels=labels, ylabel="比例", title="分AC/方向失败与过期")
    axes[0].tick_params(axis="x", rotation=35); axes[0].legend(); axes[0].grid(axis="y", alpha=.22)
    realtime = [("VO", "UL", 20), ("VO", "DL", 20), ("VI", "UL", 50), ("VI", "DL", 50)]
    timely = [np.mean([r[f"{direction}_{ac}_on_time_rate"] for r in per_seed]) for ac, direction, _ in realtime]
    axes[1].bar(np.arange(4), timely, color=[COLORS[ac] for ac, _, _ in realtime])
    axes[1].set(xticks=np.arange(4), xticklabels=[f"{ac}-{d}\n≤{deadline}ms" for ac,d,deadline in realtime],
                ylim=(0,1.03), ylabel="及时率", title="VO/VI参考期限及时交付")
    axes[1].grid(axis="y", alpha=.22)
    fig.savefig(FIG / "failure_expiry_and_timely.png", dpi=220); plt.close(fig)

    # Arrival-process competition versus full buffer.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    axes[0].bar([str(s) for s in seeds], [r["mean_active_stas"] for r in per_seed], color="#4c72b0")
    axes[0].set(ylim=(0,41), xlabel="seed", ylabel="STA数", title="随机到达：同时待发STA均值")
    q_values = [np.mean([r[f"{ac}_queue_nonempty_fraction"] for r in per_seed]) for ac in ACS]
    axes[1].bar(ACS, q_values, color=[COLORS[ac] for ac in ACS])
    axes[1].set(ylim=(0,1.03), ylabel="非空时间比例", title="随机到达：节点-AC队列非空")
    for ax in axes: ax.grid(axis="y", alpha=.22)
    fig.savefig(FIG / "arrival_competition_diagnostics.png", dpi=220); plt.close(fig)

    full_summary = json.loads((FULL / "summary.json").read_text(encoding="utf-8"))
    totals = [r["total_throughput_mbps"] for r in per_seed]
    busy = [r["busy_fraction"] for r in per_seed]
    summary = {
        "validation": "PASS" if all(item["status"] == "PASS" for item in validation.values()) else "FAIL",
        "seeds": list(seeds), "duration_s": 20,
        "total_throughput_mbps_mean": float(np.mean(totals)),
        "total_throughput_mbps_std": float(np.std(totals, ddof=1)),
        "busy_fraction_mean": float(np.mean(busy)),
        "mean_active_stas": float(np.mean([r["mean_active_stas"] for r in per_seed])),
        "maximum_active_stas": max(r["max_active_stas"] for r in per_seed),
        "fullbuffer_total_throughput_mbps_mean": full_summary["total_throughput_mbps_mean"],
        "fullbuffer_busy_fraction_mean": full_summary["busy_fraction_mean"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
