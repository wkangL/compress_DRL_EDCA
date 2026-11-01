#!/usr/bin/env python3
"""Analyze formal fixed-action runs, Pareto tradeoffs and application/MAC pipeline."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config
from sim.edca.cw_growth import Growth

RUN = ROOT / "results/raw/tgax_fixed_feasibility_formal_20s_seed44_45_20261028"
OUT = ROOT / "results/summary/tgax_fixed_feasibility_formal_20261028"
FIG = ROOT / "results/figures/tgax_fixed_feasibility_formal_20261028"
ACS = ("vo", "vi", "be", "bk")
AC_UPPER = ("VO", "VI", "BE", "BK")
LABELS = {"default_native": "Default", "p3_native": "P3+BEB", "p3_c_late24": "P3+CW",
          "p0_c_strong": "P0+CW-strong", "p0_c_late24": "P0+CW-late24"}

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False})


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = []
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def blocks(path: Path) -> list[dict]:
    output = []; current = {"M": [], "U": []}
    with (path / "events.log").open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            f = line.split()
            if not f: continue
            if f[0] in current: current[f[0]].append(tuple(map(float, f[1:])))
            elif f[0] == "END": output.append(current); current = {"M": [], "U": []}
    return output


def p95(path: Path, ac: int, duration: float) -> tuple[float | None, int]:
    samples = []
    with (path / "events.log").open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            f = line.split()
            if f and f[0] == "Y" and 1.0 < float(f[1]) <= 1.0 + duration and int(float(f[3])) == ac:
                samples.append(float(f[4]))
    return (float(np.percentile(samples, 95)), len(samples)) if samples else (None, 0)


def mac_ledger(path: Path) -> dict[int, dict[str, float]]:
    result = {ac: {"arrived_packets": 0, "acked_packets": 0, "failed_packets": 0,
                   "arrived_bytes": 0, "acked_bytes": 0, "failed_bytes": 0,
                   "attempts": 0, "retries": 0} for ac in range(4)}
    for block in blocks(path)[1:]:  # block 0 is the common one-second Default traffic warmup
        for row in block["M"]:
            ac = int(row[1]); target = result[ac]
            for key, index in (("arrived_packets", 2), ("acked_packets", 3), ("failed_packets", 4),
                               ("arrived_bytes", 7), ("acked_bytes", 8), ("failed_bytes", 9),
                               ("attempts", 10), ("retries", 11)):
                target[key] += int(row[index])
    return result


def pareto(rows: list[dict]) -> set[str]:
    result = set()
    for current in rows:
        dominated = False
        for other in rows:
            if current is other: continue
            no_worse = (other["application_throughput_mbps"] >= current["application_throughput_mbps"] and
                        other["combined_delay_normalized"] <= current["combined_delay_normalized"] and
                        other["jain"] >= current["jain"])
            strict = (other["application_throughput_mbps"] > current["application_throughput_mbps"] or
                      other["combined_delay_normalized"] < current["combined_delay_normalized"] or
                      other["jain"] > current["jain"])
            if no_worse and strict: dominated = True; break
        if not dominated: result.add(current["candidate"])
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    experiment = json.loads((RUN / "experiment_config.json").read_text(encoding="utf-8"))
    duration = float(experiment["duration_s"]); candidates = experiment["candidates"]
    per_seed = []; pipeline = []
    for seed in experiment["seeds"]:
        for name, spec in candidates.items():
            path = RUN / f"seed_{seed}" / name
            summary = json.loads((path / "fixed_action_summary.json").read_text(encoding="utf-8"))
            generated = np.asarray(summary["application_generated_bytes"], dtype=float)
            submitted = np.asarray(summary["application_submitted_bytes"], dtype=float)
            received = np.asarray(summary["application_received_bytes"], dtype=float)
            ledger = mac_ledger(path); delays = [p95(path, ac, duration) for ac in range(4)]
            row = {"seed": seed, "candidate": name, "label": LABELS[name], "action": json.dumps(spec["action"]),
                   "custom": spec["custom"], "application_throughput_mbps": received.sum()*8/duration/1e6,
                   "application_satisfaction": received.sum()/generated.sum(),
                   "socket_submission_fraction": submitted.sum()/generated.sum(),
                   "jain": summary["jain_demand_satisfaction"], "vo_p95_ms": delays[0][0],
                   "vi_p95_ms": delays[1][0], "vo_delay_samples": delays[0][1], "vi_delay_samples": delays[1][1],
                   "combined_delay_normalized": 0.5*(delays[0][0]/20 + delays[1][0]/50),
                   "vo_timely": summary["vo_on_time_delivery_rate"], "vi_timely": summary["vi_on_time_delivery_rate"]}
            for ac, label in enumerate(ACS):
                target = ledger[ac]
                # Final-drop counters in the STEP summary include packets already at risk
                # when the measured window begins.  Weight the two directions by their
                # complete risk populations instead of dividing measurement-window drops
                # by measurement-window arrivals (which can exceed one after warmup).
                risk = sum(float(summary.get(f"{direction}_{label}_risk_packets", 0) or 0)
                           for direction in ("ul", "dl"))
                final_drops = sum(float(summary.get(f"{direction}_{label}_final_drop_rate", 0) or 0) *
                                  float(summary.get(f"{direction}_{label}_risk_packets", 0) or 0)
                                  for direction in ("ul", "dl"))
                row[f"{label}_application_throughput_mbps"] = received[ac]*8/duration/1e6
                row[f"{label}_mac_success_throughput_mbps"] = target["acked_bytes"]*8/duration/1e6
                row[f"{label}_application_satisfaction"] = received[ac]/generated[ac] if generated[ac] else None
                row[f"{label}_final_failure_rate"] = final_drops/risk if risk else None
                row[f"{label}_retransmission_rate"] = target["retries"]/target["attempts"] if target["attempts"] else None
                pipeline.append({"seed": seed, "candidate": name, "ac": label.upper(),
                                 "generated_bytes": generated[ac], "socket_submitted_bytes": submitted[ac],
                                 "mac_admitted_mpdu_bytes": target["arrived_bytes"],
                                 "mac_successfully_acked_mpdu_bytes": target["acked_bytes"],
                                 "application_received_payload_bytes": received[ac],
                                 "socket_submission_over_generated": submitted[ac]/generated[ac] if generated[ac] else None,
                                 "application_received_over_generated": received[ac]/generated[ac] if generated[ac] else None,
                                 "mac_note": "MAC byte counters include transport/IP payload framing and reverse TCP ACK MPDUs; not byte-identical to application payload"})
            pipeline.append({"seed": seed, "candidate": name, "ac": "ALL",
                             "generated_bytes": generated.sum(), "socket_submitted_bytes": submitted.sum(),
                             "mac_admitted_mpdu_bytes": sum(v["arrived_bytes"] for v in ledger.values()),
                             "mac_successfully_acked_mpdu_bytes": sum(v["acked_bytes"] for v in ledger.values()),
                             "application_received_payload_bytes": received.sum(),
                             "socket_submission_over_generated": submitted.sum()/generated.sum(),
                             "application_received_over_generated": received.sum()/generated.sum(),
                             "mac_note": "Application stages are comparable only in aggregate; per-AC submitted/received can exceed same-window generated because warmup/backlog crosses the boundary. MAC bytes include protocol overhead and reverse ACKs."})
            per_seed.append(row)
    write_csv(OUT / "formal_per_seed.csv", per_seed); write_csv(OUT / "demand_mac_delivery_pipeline.csv", pipeline)

    means = []
    for name in candidates:
        group = [r for r in per_seed if r["candidate"] == name]
        row = {"candidate": name, "label": LABELS[name], "action": group[0]["action"], "custom": group[0]["custom"]}
        for key in per_seed[0]:
            if key in {"seed", "candidate", "label", "action", "custom"}: continue
            values = [r[key] for r in group if r[key] is not None]
            row[key] = float(np.mean(values)) if values else None
        means.append(row)
    front = pareto(means); default = next(r for r in means if r["candidate"] == "default_native")
    directions = {"application_throughput_mbps": 1, "combined_delay_normalized": -1, "vo_p95_ms": -1,
                  "vi_p95_ms": -1, "jain": 1, "vo_timely": 1, "vi_timely": 1}
    for row in means:
        row["pareto"] = row["candidate"] in front
        for metric, direction in directions.items():
            raw = (row[metric]/default[metric]-1)*100 if default[metric] else None
            row[f"{metric}_change_percent"] = raw
            row[f"{metric}_improvement_percent"] = raw*direction if raw is not None else None
        row["strictly_dominates_default"] = (row["application_throughput_mbps"] > default["application_throughput_mbps"] and
            row["combined_delay_normalized"] < default["combined_delay_normalized"] and row["jain"] > default["jain"])
    write_csv(OUT / "formal_mean_and_percentages.csv", means)

    # Figure 1: three-objective distribution.  Delay definition was frozen before formal seeds.
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    fairness = np.asarray([r["jain"] for r in means]); scatter = ax.scatter(
        [r["application_throughput_mbps"] for r in means], [r["combined_delay_normalized"] for r in means],
        c=fairness, cmap="viridis", s=[160 if r["candidate"] == "default_native" else 105 for r in means],
        edgecolors=["red" if r["candidate"] == "default_native" else "black" for r in means], linewidths=1.3)
    for row in means:
        ax.annotate(row["label"] + (" ★" if row["pareto"] else ""),
                    (row["application_throughput_mbps"], row["combined_delay_normalized"]), xytext=(5,5),
                    textcoords="offset points", fontsize=8)
    ax.set(xlabel="BSS0应用有效载荷吞吐量 (Mbps)", ylabel="综合时延 0.5×(VO p95/20 + VI p95/50)",
           title="固定EDCA/CW：吞吐—时延—公平性分布（★为三目标Pareto前沿）")
    ax.grid(alpha=.22); fig.colorbar(scatter, ax=ax, label="Jain需求满足公平性")
    fig.savefig(FIG / "throughput_delay_fairness_pareto.png", dpi=230); plt.close(fig)

    # Figure 2: representative core metrics.
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)
    metrics = (("application_throughput_mbps", "总应用有效吞吐 (Mbps)"), ("vo_p95_ms", "VO MAC服务时延p95 (ms)"),
               ("vi_p95_ms", "VI MAC服务时延p95 (ms)"), ("jain", "Jain需求满足公平性"))
    colors = ["#d62728" if r["candidate"] == "default_native" else "#3973b7" for r in means]
    for ax, (metric, title) in zip(axes.flat, metrics):
        ax.bar([r["label"] for r in means], [r[metric] for r in means], color=colors)
        ax.set_title(title); ax.tick_params(axis="x", rotation=22); ax.grid(axis="y", alpha=.22)
    fig.savefig(FIG / "representative_core_metrics.png", dpi=230); plt.close(fig)

    # Figure 3: application payload throughput by AC.
    fig, ax = plt.subplots(figsize=(10.6, 5.5), constrained_layout=True); x=np.arange(len(means)); width=.19
    for i, ac in enumerate(ACS):
        ax.bar(x+(i-1.5)*width, [r[f"{ac}_application_throughput_mbps"] for r in means], width, label=ac.upper())
    ax.set(xticks=x, xticklabels=[r["label"] for r in means], ylabel="应用有效载荷吞吐 (Mbps)",
           title="各AC应用有效吞吐：检查高优先级收益是否牺牲BE/BK")
    ax.tick_params(axis="x", rotation=18); ax.grid(axis="y", alpha=.22); ax.legend()
    fig.savefig(FIG / "application_throughput_by_ac.png", dpi=230); plt.close(fig)

    # Figure 4: actual CW trajectories for selected mechanisms.
    config = load_config(ROOT / "configs/tgax_ddqn_reward_v3.yaml"); profiles=json.loads((ROOT/config["action"]["profiles"]).read_text())
    chosen = ["default_native", "p0_c_late24", "p0_c_strong", "p3_c_late24"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax, ac in zip(axes.flat, AC_UPPER):
        for name in chosen:
            spec=candidates[name]; action=spec["action"]; profile=0 if action[0]==16 else action[0]; p=profiles[profile][ac]
            if spec["custom"]:
                params=(config["action"]["alpha"][action[1]],config["action"]["beta1"][action[2]],
                        config["action"]["beta2"][action[3]],config["action"]["gamma"][action[4]])
                trajectory=Growth(*params).table(p["CWmin"],p["CWmax"],10)
            else:
                trajectory=[p["CWmin"]]+[min(p["CWmax"],(p["CWmin"]+1)*2**n-1) for n in range(1,11)]
            ax.plot(range(11),trajectory,"o-",label=LABELS[name],markersize=3)
        ax.set(title=ac,xlabel="累计失败/重试阶段",ylabel="CW",yscale="symlog",xticks=range(11));ax.grid(alpha=.22);ax.legend(fontsize=7)
    fig.savefig(FIG / "representative_cw_trajectories.png", dpi=230); plt.close(fig)

    manifest = {"formal_run_status": json.loads((RUN/"status.json").read_text()),
                "combined_delay_formula": "0.5*(VO aggregate raw-sample p95/20ms + VI aggregate raw-sample p95/50ms)",
                "pareto_candidates": sorted(front), "strict_default_dominators": [r["candidate"] for r in means if r["strictly_dominates_default"]],
                "screen_seeds_reused_for_exploration_only": [41,42], "formal_seeds": [44,45],
                "future_ddqn_holdout_warning": "Seeds 41,42,44,45 are no longer eligible as final held-out DDQN test seeds."}
    (OUT/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
