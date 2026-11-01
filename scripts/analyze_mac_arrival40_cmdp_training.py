#!/usr/bin/env python3
"""Summarize the single-BSS high-Gaming CMDP training run."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/raw/mac_arrival40_gaming160_cmdp_seed44_400step_20261029"
SUMMARY = ROOT / "results/summary/mac_arrival40_gaming160_cmdp_seed44_400step_20261030"
FIGURES = ROOT / "results/figures/mac_arrival40_gaming160_cmdp_seed44_400step_20261030"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, data: list[dict]) -> None:
    if not data:
        return
    keys = list(data[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(data)


def moving(values: np.ndarray, n: int = 20) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(values) >= n:
        out[n - 1:] = np.convolve(values, np.ones(n) / n, mode="valid")
    return out


def finite(values) -> bool:
    return all(math.isfinite(float(v)) for v in values)


def controlled_delay_ccdf(folder: Path, start_s: float = 1.0, end_s: float = 21.0) -> list[dict]:
    """Build exact CCDF rows from successful MPDUs in the 20 controlled seconds."""
    grouped=defaultdict(list)
    for line in (folder/"events.log").read_text(encoding="utf-8").splitlines():
        if not line.startswith("Y "):
            continue
        fields=line.split(); time=float(fields[1])
        if not (start_s <= time < end_s):
            continue
        direction=("UL","DL")[int(fields[2])]
        ac=("VO","VI","BE","BK")[int(fields[3])]
        grouped[direction,ac].append(float(fields[4]))
    output=[]
    for (direction,ac), values in grouped.items():
        ordered=np.sort(np.asarray(values,dtype=float)); n=len(ordered)
        for i,value in enumerate(ordered):
            output.append({"scope":"BSS","bss":"0","direction":direction,"ac":ac,
                           "delay_ms":value,"ccdf":(n-i)/n})
    return output


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    steps = read_csv(RUN / "training_steps.csv")
    losses = read_csv(RUN / "losses.csv")
    curve = read_csv(RUN / "validation/learning_curve.csv")
    default = json.loads((RUN / "held_out/seed_47/default/evaluation_summary.json").read_text(encoding="utf-8"))
    learned = json.loads((RUN / "held_out/seed_47/selected/evaluation_summary.json").read_text(encoding="utf-8"))

    # Training diagnostics.
    sx = np.asarray([int(r["step"]) for r in steps])
    ux = np.asarray([int(r["update"]) for r in losses])
    reward = np.asarray([float(r["reward"]) for r in steps])
    utility = np.asarray([float(r["cmdp_utility"]) for r in steps])
    costs = np.asarray([json.loads(r["cmdp_costs"]) for r in steps], dtype=float)
    lambdas = np.asarray([json.loads(r["cmdp_multipliers"]) for r in steps], dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)
    axes[0, 0].plot(sx, reward, alpha=.22, lw=.7); axes[0, 0].plot(sx, moving(reward), label="Lagrangian reward")
    axes[0, 0].plot(sx, moving(utility), label="utility"); axes[0, 0].legend(fontsize=8)
    axes[0, 0].set(title="Training objective", xlabel="environment step")
    loss = np.asarray([float(r["td_loss"]) for r in losses])
    axes[0, 1].plot(ux, loss, alpha=.25); axes[0, 1].plot(ux, moving(loss), lw=2)
    axes[0, 1].set(title="Q loss", xlabel="gradient update", ylabel="Huber loss")
    axes[0, 2].plot(ux, [float(r["q_mean"]) for r in losses], label="Online Q")
    axes[0, 2].plot(ux, [float(r["target_q_mean"]) for r in losses], label="Target Q")
    axes[0, 2].plot(ux, [float(r["td_target_mean"]) for r in losses], label="TD target", alpha=.75)
    axes[0, 2].set(title="Q-value diagnostics", xlabel="gradient update"); axes[0, 2].legend(fontsize=8)
    for i, label in enumerate(("VO miss", "VI miss", "Jain deficit", "BE/BK deficit")):
        axes[1, 0].plot(sx, moving(costs[:, i]), label=label)
        axes[1, 1].plot(sx, lambdas[:, i], label=label)
    axes[1, 0].set(title="CMDP costs (20-step mean)", xlabel="environment step"); axes[1, 0].legend(fontsize=7)
    axes[1, 1].set(title="Lagrange multipliers", xlabel="environment step"); axes[1, 1].legend(fontsize=7)
    axes[1, 2].plot(ux, [float(r["grad_norm"]) for r in losses], label="gradient norm")
    axes[1, 2].plot(ux, [float(r["td_error_abs_mean"]) for r in losses], label="mean |TD error|")
    axes[1, 2].set(title="Optimization stability", xlabel="gradient update"); axes[1, 2].legend(fontsize=8)
    for ax in axes.flat: ax.grid(alpha=.22)
    fig.savefig(FIGURES / "training_diagnostics.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.plot(ux, loss, color="#7aa6c2", alpha=.42, lw=.8, label="per-update Q loss")
    ax.plot(ux, moving(loss), color="#c43c39", lw=2.1, label="20-update moving mean")
    ax.set(title="TCN-DDQN Q loss", xlabel="gradient update", ylabel="Huber TD loss")
    ax.grid(alpha=.24); ax.legend()
    fig.savefig(FIGURES / "q_loss.png", dpi=220); plt.close(fig)

    # Validation learning curve.
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 4.0), constrained_layout=True)
    for seed, color in ((45, "#2878b5"), (46, "#e07a2d")):
        rows = [r for r in curve if int(r["eval_seed"]) == seed]
        x = [int(r["training_steps"]) for r in rows]
        axes[0].plot(x, [float(r["learned_cmdp_utility"]) for r in rows], "o-", color=color, label=f"seed {seed}")
        axes[1].plot(x, [100*float(r["learned_vo_on_time_delivery_rate"]) for r in rows], "o-", color=color, label=f"VO s{seed}")
        axes[1].plot(x, [100*float(r["learned_vi_on_time_delivery_rate"]) for r in rows], "s--", color=color, label=f"VI s{seed}")
        axes[2].plot(x, [float(r["learned_jain_demand_satisfaction"]) for r in rows], "o-", color=color, label=f"seed {seed}")
    axes[0].set(title="Independent validation utility", ylabel="utility")
    axes[1].set(title="VO/VI timely delivery", ylabel="percent")
    axes[2].set(title="Demand-normalized Jain", ylabel="Jain index")
    for ax in axes: ax.set(xlabel="training step", xticks=[100,200,300,400]); ax.grid(alpha=.22); ax.legend(fontsize=7)
    fig.savefig(FIGURES / "validation_learning_curve.png", dpi=220); plt.close(fig)

    # Held-out comparison and action distribution.
    metrics = [
        ("CMDP utility", "cmdp_utility_mean"), ("VO timely rate", "vo_on_time_delivery_rate"),
        ("VI timely rate", "vi_on_time_delivery_rate"), ("Jain", "jain_demand_satisfaction"),
        ("VO satisfaction", "vo_demand_satisfaction"), ("VI satisfaction", "vi_demand_satisfaction"),
        ("BE satisfaction", "be_demand_satisfaction"), ("BK satisfaction", "bk_demand_satisfaction"),
        ("UL VO p95 ms", "ul_vo_delay_p95_ms"), ("UL VI p95 ms", "ul_vi_delay_p95_ms"),
        ("UL VO throughput Mbps", "ul_vo_throughput_mbps"), ("UL VI throughput Mbps", "ul_vi_throughput_mbps"),
        ("UL BE throughput Mbps", "ul_be_throughput_mbps"), ("UL BK throughput Mbps", "ul_bk_throughput_mbps"),
    ]
    comparison = []
    for name, key in metrics:
        a, b = float(default[key]), float(learned[key])
        comparison.append({"metric": name, "default": a, "learned": b,
                           "change_percent": 100 * (b - a) / a if a else math.nan})
    write_csv(SUMMARY / "held_out_seed47_comparison.csv", comparison)
    write_csv(SUMMARY / "validation_learning_curve.csv", curve)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), constrained_layout=True)
    labels = ["VO timely", "VI timely", "Jain", "VO sat.", "VI sat.", "BE sat.", "BK sat."]
    keys = ["vo_on_time_delivery_rate", "vi_on_time_delivery_rate", "jain_demand_satisfaction",
            "vo_demand_satisfaction", "vi_demand_satisfaction", "be_demand_satisfaction", "bk_demand_satisfaction"]
    x = np.arange(len(keys)); w=.36
    axes[0].bar(x-w/2, [default[k] for k in keys], w, label="Default")
    axes[0].bar(x+w/2, [learned[k] for k in keys], w, label="CMDP-DDQN")
    axes[0].set(title="Held-out seed 47: service outcomes", xticks=x, xticklabels=labels, ylim=(0,1)); axes[0].tick_params(axis='x', rotation=35); axes[0].legend()
    tkeys = ["ul_vo_throughput_mbps","ul_vi_throughput_mbps","ul_be_throughput_mbps","ul_bk_throughput_mbps"]
    x2=np.arange(4)
    axes[1].bar(x2-w/2,[default[k] for k in tkeys],w,label="Default")
    axes[1].bar(x2+w/2,[learned[k] for k in tkeys],w,label="CMDP-DDQN")
    axes[1].set(title="Held-out seed 47: UL throughput", ylabel="Mbps", xticks=x2, xticklabels=["VO","VI","BE","BK"]); axes[1].legend()
    for ax in axes: ax.grid(axis="y", alpha=.22)
    fig.savefig(FIGURES / "held_out_core_metrics.png", dpi=220); plt.close(fig)

    actions = learned["actions"]
    names=("Profile","alpha","beta1","beta2","gamma")
    physical_values=(None, [1,2,3], [1.6,2.0], [2.0,2.4,2.8], [5,6,7])
    fig, axes=plt.subplots(1,5,figsize=(13.5,3.2),constrained_layout=True)
    action_rows=[]
    for i,(name,ax) in enumerate(zip(names,axes)):
        count=Counter(a[i] for a in actions); xs=sorted(count)
        labels=[f"P{v}" if i==0 else str(physical_values[i][v]) for v in xs]
        ax.bar(labels,[count[v] for v in xs]); ax.set(title=name,ylabel="count" if i==0 else None)
        for v in xs:
            value=f"P{v}" if i==0 else physical_values[i][v]
            action_rows.append({"branch":name,"index":v,"physical_value":value,
                                "count":count[v],"fraction":count[v]/len(actions)})
    fig.savefig(FIGURES / "held_out_action_distribution.png", dpi=220); plt.close(fig)
    write_csv(SUMMARY / "held_out_action_distribution.csv", action_rows)

    # Successful-MPDU MAC service-delay CCDF, using the raw ns-3 samples.
    ccdf_default = controlled_delay_ccdf(RUN / "held_out/seed_47/default")
    ccdf_learned = controlled_delay_ccdf(RUN / "held_out/seed_47/selected")
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), constrained_layout=True)
    quantile_rows=[]
    for ax, (ac, direction) in zip(axes.flat, (("VO", "UL"), ("VO", "DL"), ("VI", "UL"), ("VI", "DL"))):
        panel_values=[]
        for data, label, color in ((ccdf_default, "Default", "#2878b5"),
                                   (ccdf_learned, "CMDP-DDQN", "#d95f02")):
            current = [r for r in data if r["scope"] == "BSS" and r["bss"] == "0"
                       and r["ac"] == ac and r["direction"] == direction]
            if current:
                values=np.asarray([float(r["delay_ms"]) for r in current])
                panel_values.append(values)
                ax.plot(values,
                        [max(float(r["ccdf"]), 1e-5) for r in current],
                        lw=1.6, label=label, color=color)
                for q in (50,80,90,95,98,99):
                    quantile_rows.append({"ac":ac,"direction":direction,"policy":label,
                                          "percentile":q,"delay_ms":float(np.percentile(values,q))})
        ax.axvline(20 if ac == "VO" else 50, color="#666", ls="--", lw=1,
                   label=f"{20 if ac == 'VO' else 50} ms deadline")
        # Stop at P99: the smallest DL group has only about 500 samples, so deeper
        # tails are visually unstable and should remain an audit-only figure.
        ticks=[1,.5,.2,.1,.05,.02,.01]
        ax.set_yscale("log"); ax.set_ylim(.01, 1.05); ax.set_yticks(ticks)
        ax.set_yticklabels(["P0","P50","P80","P90","P95","P98","P99"])
        if panel_values:
            ax.set_xlim(0, max(np.percentile(v,99) for v in panel_values)*1.15)
        ax.set(title=f"{ac} {direction}", xlabel="MAC enqueue-to-successful-ACK service delay (ms)",
               ylabel="completed delay percentile")
        ax.grid(True, which="both", alpha=.22); ax.legend(fontsize=8)
    fig.savefig(FIGURES / "held_out_vo_vi_delay_ccdf.png", dpi=220); plt.close(fig)
    write_csv(SUMMARY / "held_out_vo_vi_delay_percentiles.csv",quantile_rows)

    # Preserve the very deep tail as a separate audit plot rather than using it
    # as the report's primary comparison.
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), constrained_layout=True)
    for ax, (ac, direction) in zip(axes.flat, (("VO", "UL"), ("VO", "DL"), ("VI", "UL"), ("VI", "DL"))):
        for data, label, color in ((ccdf_default, "Default", "#2878b5"),
                                   (ccdf_learned, "CMDP-DDQN", "#d95f02")):
            current=[r for r in data if r["scope"]=="BSS" and r["bss"]=="0"
                     and r["ac"]==ac and r["direction"]==direction]
            if current:
                ax.plot([float(r["delay_ms"]) for r in current],
                        [max(float(r["ccdf"]),1e-5) for r in current],lw=1.4,label=label,color=color)
        ax.axvline(20 if ac=="VO" else 50,color="#666",ls="--",lw=1)
        ax.set_yscale("log"); ax.set_ylim(1e-5,1.05); ax.set_xlim(left=0)
        ax.set(title=f"{ac} {direction}",xlabel="MAC enqueue-to-successful-ACK delay (ms)",
               ylabel="CCDF  P(delay > x)")
        ax.grid(True,which="both",alpha=.22); ax.legend(fontsize=8)
    fig.savefig(FIGURES / "held_out_vo_vi_delay_ccdf_full_tail.png",dpi=220); plt.close(fig)

    # AC-level CCDFs with UL/DL combined using the *actual* successful-packet
    # counts.  The stored delay points are bounded per-direction samples, so a
    # plain concatenation would over-weight the much smaller DL population.
    combined_rows=[]
    for ac in ("VO","VI"):
        fig, ax=plt.subplots(figsize=(8.2,5.0),constrained_layout=True)
        all_curves=[]
        for policy,folder,data,label,color in (
            ("default","default",ccdf_default,"Default","#2878b5"),
            ("CMDP-DDQN","selected",ccdf_learned,"CMDP-DDQN","#d95f02")):
            samples={}; weights={}
            for direction in ("UL","DL"):
                samples[direction]=np.sort(np.asarray([float(r["delay_ms"]) for r in data
                    if r["scope"]=="BSS" and r["bss"]=="0" and r["ac"]==ac
                    and r["direction"]==direction]))
                weights[direction]=len(samples[direction])
            grid=np.unique(np.concatenate([v for v in samples.values() if len(v)]))
            total=sum(weights[d] for d in samples if len(samples[d]))
            survival=np.zeros(len(grid))
            for direction,values in samples.items():
                if len(values):
                    survival += weights[direction]*(len(values)-np.searchsorted(values,grid,side="left"))/len(values)
            survival/=max(total,1)
            ax.plot(grid,np.maximum(survival,1e-5),lw=2,label=label,color=color)
            all_curves.append((grid,survival))
            for q in (50,80,90,95,98,99):
                target=1-q/100
                idx=np.flatnonzero(survival<=target)
                value=float(grid[idx[0]]) if len(idx) else float(grid[-1])
                combined_rows.append({"ac":ac,"policy":policy,"percentile":q,
                                      "delay_ms":value,"ul_success_weight":weights["UL"],
                                      "dl_success_weight":weights["DL"]})
        ticks=[1,.5,.2,.1,.05,.02,.01]
        ax.axvline(20 if ac=="VO" else 50,color="#666",ls="--",lw=1.2,
                   label=f"{20 if ac=='VO' else 50} ms deadline")
        ax.set_yscale("log"); ax.set_ylim(.01,1.05); ax.set_yticks(ticks)
        ax.set_yticklabels(["P0","P50","P80","P90","P95","P98","P99"])
        p99=[]
        for grid,survival in all_curves:
            idx=np.flatnonzero(survival<=.01); p99.append(grid[idx[0]] if len(idx) else grid[-1])
        ax.set_xlim(0,max(p99)*1.15)
        ax.set(title=f"{ac} MAC service delay CCDF (UL+DL packet-weighted)",
               xlabel="MAC enqueue-to-successful-ACK service delay (ms)",
               ylabel="completed delay percentile")
        ax.grid(True,which="both",alpha=.23); ax.legend()
        fig.savefig(FIGURES/f"held_out_{ac.lower()}_delay_ccdf_combined.png",dpi=220); plt.close(fig)
    write_csv(SUMMARY/"held_out_vo_vi_delay_percentiles_combined.csv",combined_rows)

    stability = {
        "steps": len(steps), "updates": len(losses),
        "all_finite": finite([r[k] for r in losses for k in ("td_loss","q_mean","q_max","target_q_max","grad_norm")]),
        "td_loss_first20_mean": float(np.mean(loss[:20])), "td_loss_last20_mean": float(np.mean(loss[-20:])),
        "q_abs_max": max(abs(float(r["q_min"])) for r in losses) if losses else None,
        "q_positive_max": max(float(r["q_max"]) for r in losses) if losses else None,
        "target_q_max": max(float(r["target_q_max"]) for r in losses) if losses else None,
        "grad_norm_max": max(float(r["grad_norm"]) for r in losses) if losses else None,
        "final_lambdas": lambdas[-1].tolist(), "selected_step": 400,
        "held_out_generated_demand_paired": False,
    }
    (SUMMARY / "numerical_stability.json").write_text(json.dumps(stability, indent=2), encoding="utf-8")
    print(json.dumps(stability, indent=2))


if __name__ == "__main__":
    main()
