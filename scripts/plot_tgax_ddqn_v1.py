"""Create auditable summaries and figures for the first TGax DDQN run."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/raw/tgax_ddqn_v1_small_seed41_r4_20261023"
SUMMARY = ROOT / "results/summary/tgax_ddqn_v1_fixed_assoc_20261023"
FIGURES = ROOT / "results/figures/tgax_ddqn_v1_small_seed41_r4_20261023"
SUMMARY.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


result = json.loads((RUN / "result.json").read_text(encoding="utf-8"))
default, learned = result["default"], result["learned"]
for side, summary in (("default", default), ("learned", learned)):
    step_rows = rows(RUN / f"eval_{side}" / "evaluation_steps.csv")
    for ac in ("vo", "vi", "be", "bk"):
        for direction in ("ul", "dl"):
            prefix=f"{direction}_{ac}"
            arrived=sum(int(float(r[f"{prefix}_arrived"])) for r in step_rows)
            attempts=sum(int(float(r[f"{prefix}_tx_attempts"])) for r in step_rows)
            if summary[f"{prefix}_successful_samples"] == 0:
                summary[f"{prefix}_delay_p95_ms"] = None
            # No MAC admission/attempt sample is undefined, not a perfect zero rate.
            if arrived == 0:
                summary[f"{prefix}_final_drop_rate"] = None
            if attempts == 0:
                summary[f"{prefix}_retransmission_rate"] = None
(SUMMARY / "evaluation_result_corrected.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

# Positive improvement always means better; undefined values stay blank.
comparison = []
for ac in ("vo", "vi", "be", "bk"):
    for direction in ("ul", "dl"):
        for metric, unit, lower in (
            ("delay_p95_ms", "ms", True),
            ("throughput_mbps", "Mbps", False),
            ("final_drop_rate", "fraction", True),
            ("retransmission_rate", "fraction", True),
        ):
            key = f"{direction}_{ac}_{metric}"
            old, new = default[key], learned[key]
            improvement = None
            if old is not None and new is not None and old != 0:
                improvement = (old - new) / old * 100 if lower else (new - old) / old * 100
            comparison.append({"ac": ac.upper(), "direction": direction.upper(), "metric": metric,
                               "unit": unit, "default": old, "learned": new,
                               "improvement_percent_positive_is_better": improvement})
for metric in ("reward_mean", "jain_demand_satisfaction"):
    old, new = default[metric], learned[metric]
    comparison.append({"ac": "ALL", "direction": "BSS0", "metric": metric, "unit": "score",
                       "default": old, "learned": new,
                       "improvement_percent_positive_is_better": (new-old)/old*100 if old else None})
with (SUMMARY / "evaluation_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(comparison[0])); writer.writeheader(); writer.writerows(comparison)

# Demand-sequence equality audit.
demand = {}
for side in ("default", "learned"):
    data = rows(RUN / f"eval_{side}" / "evaluation_steps.csv")
    demand[side] = [json.loads(r["application_generated_bytes"]) for r in data]
demand_audit = {"equal_each_step": demand["default"] == demand["learned"], **demand}
(SUMMARY / "evaluation_demand_audit.json").write_text(json.dumps(demand_audit, indent=2), encoding="utf-8")

# EDCA isolation audit: after common warm-up, neighbors must always be native defaults.
native = {"VO": (2,3,7,2080), "VI": (2,7,15,4096),
          "BE": (3,15,1023,0), "BK": (7,15,1023,0)}
edca = rows(RUN / "eval_learned" / "edca_step_runtime.csv")
neighbor_bad = []
target_changed = 0
for r in edca:
    actual = tuple(int(float(r[k])) for k in ("aifsn", "cwmin", "cwmax", "txop_us"))
    if int(r["bss"]) > 0 and actual != native[r["ac"]]: neighbor_bad.append(r)
    if int(r["bss"]) == 0 and float(r["time_s"]) >= 1.0 and actual != native[r["ac"]]: target_changed += 1
isolation = {"neighbor_rows_checked": sum(int(r["bss"]) > 0 for r in edca),
             "neighbor_nondefault_rows": len(neighbor_bad), "target_nondefault_rows": target_changed}
(SUMMARY / "evaluation_action_isolation.json").write_text(json.dumps(isolation, indent=2), encoding="utf-8")

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25,
                     "figure.dpi": 140, "savefig.dpi": 220})

# Training diagnostics.
steps = rows(RUN / "training_steps.csv")
loss = rows(RUN / "losses.csv")
fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
x = np.array([int(r["step"]) for r in steps]); reward = np.array([float(r["reward"]) for r in steps])
axes[0,0].plot(x, reward, marker="o"); axes[0,0].set(title="Training reward", xlabel="Environment step", ylabel="Reward")
u = np.array([int(r["update"]) for r in loss]); qloss = np.array([float(r["td_loss"]) for r in loss])
smooth = np.convolve(qloss, np.ones(5)/5, mode="valid")
axes[0,1].plot(u, qloss, alpha=.35, label="raw"); axes[0,1].plot(u[4:], smooth, label="MA(5)")
axes[0,1].set(title="TD loss", xlabel="Gradient update", ylabel="Huber loss"); axes[0,1].legend()
axes[1,0].plot(u, [float(r["q_mean"]) for r in loss], label="online Q")
axes[1,0].plot(u, [float(r["target_q_mean"]) for r in loss], label="target Q")
axes[1,0].plot(u, [float(r["td_target_mean"]) for r in loss], label="TD target", alpha=.8)
axes[1,0].set(title="Q diagnostics", xlabel="Gradient update", ylabel="Value"); axes[1,0].legend()
branch_names = ["Profile", "alpha", "beta1", "beta2", "gamma"]
for i, item in enumerate(result["training_action_distribution"]):
    count = sum(item["counts"].values()); axes[1,1].bar(i, len(item["counts"]), label=f"{count} choices")
axes[1,1].set(title="Explored values per branch", ylabel="Distinct indices", xticks=range(5), xticklabels=branch_names)
fig.savefig(FIGURES / "training_diagnostics.png"); fig.savefig(FIGURES / "training_diagnostics.pdf"); plt.close(fig)

# Core evaluation metrics. Delay is visualized only as CCDF below; p95 stays in the table.
labels = [f"{d.upper()}-{a.upper()}" for a in ("vo","vi","be","bk") for d in ("ul","dl")]
fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
specs = [("throughput_mbps","Throughput (Mbps)"),
         ("final_drop_rate","Final drop rate"), ("retransmission_rate","Retransmission rate")]
for ax,(suffix,title) in zip(axes.flat[:3],specs):
    a=[]; b=[]
    for ac in ("vo","vi","be","bk"):
        for d in ("ul","dl"):
            a.append(default[f"{d}_{ac}_{suffix}"] if default[f"{d}_{ac}_{suffix}"] is not None else np.nan)
            b.append(learned[f"{d}_{ac}_{suffix}"] if learned[f"{d}_{ac}_{suffix}"] is not None else np.nan)
    pos=np.arange(len(labels)); w=.38
    ax.bar(pos-w/2,a,w,label="Default"); ax.bar(pos+w/2,b,w,label="Learned")
    ax.set(title=title,xticks=pos,xticklabels=labels); ax.tick_params(axis="x",rotation=35); ax.legend()
ax=axes.flat[3]; pos=np.arange(2); w=.38
ax.bar(pos-w/2,[default['reward_mean'],default['jain_demand_satisfaction']],w,label='Default')
ax.bar(pos+w/2,[learned['reward_mean'],learned['jain_demand_satisfaction']],w,label='Learned')
ax.set(title='Reward and demand-satisfaction Jain',xticks=pos,xticklabels=['Reward','Jain']);ax.legend()
fig.savefig(FIGURES / "bss0_core_metrics.png"); fig.savefig(FIGURES / "bss0_core_metrics.pdf"); plt.close(fig)

# Evaluation-only CCDF parsed from STEP Y events (exclude the 1 s common warm-up).
def delays(side: str):
    out=defaultdict(list)
    for line in (RUN/f"eval_{side}"/"events.log").read_text(encoding="utf-8").splitlines():
        f=line.split()
        if f and f[0]=="Y" and float(f[1]) > 1.0 + 1e-9:
            out[(int(float(f[2])),int(float(f[3])))].append(float(f[4]))
    return out
dd, dl = delays("default"), delays("learned")
fig, axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True,sharey=True)
for ax,(direction,ac) in zip(axes.flat,[(0,0),(1,0),(0,1),(1,1)]):
    for name,data in (("Default",dd),("Learned",dl)):
        v=np.sort(np.asarray(data[(direction,ac)])); y=(len(v)-np.arange(len(v)))/max(len(v),1)
        if len(v): ax.step(v,y,where="post",label=name)
    ax.set_yscale("log"); ax.set_ylim(1e-3,1.05); ax.set(title=f"{'UL' if direction==0 else 'DL'}-{'VO' if ac==0 else 'VI'}",
        xlabel="MAC final service delay (ms)",ylabel="CCDF"); ax.legend()
fig.savefig(FIGURES/"vo_vi_service_delay_ccdf.png"); fig.savefig(FIGURES/"vo_vi_service_delay_ccdf.pdf"); plt.close(fig)

print(json.dumps({"comparison": str(SUMMARY/'evaluation_comparison.csv'), "figures": str(FIGURES),
                  "demand_audit": demand_audit["equal_each_step"], "isolation": isolation}, indent=2))
