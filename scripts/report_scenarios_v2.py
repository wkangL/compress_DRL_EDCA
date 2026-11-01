"""Create the Stage-D baseline table and CCDF figures for calibrated scenarios v2."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT
from reporting.comparison import empirical_ccdf

ACS=("vo","vi","be","bk")


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as stream: return list(csv.DictReader(stream))


def write(path: Path,rows: list[dict]) -> None:
    with path.open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def ci95(values: list[float]) -> float:
    return 4.3026527297*np.std(values,ddof=1)/np.sqrt(len(values)) if len(values)>1 else 0.0


def main() -> None:
    run=ROOT/"results/raw/scenarios_v2_baseline_schema3_20261020"
    rows=read(run/"evaluation.csv")
    out=ROOT/"results/summary/scenarios_v2_baseline_schema3_20261020"; figs=ROOT/"results/figures/scenarios_v2_baseline_schema3_20261020"
    out.mkdir(parents=True,exist_ok=True); figs.mkdir(parents=True,exist_ok=True)
    per_seed=[]
    for row in rows:
        item=dict(row)
        case=run/f"{row['scenario']}_seed{row['eval_seed']}"
        samples=np.load(case/"delay_samples.npz")
        for ac,deadline in (("vo",20.0),("vi",50.0)):
            item[f"{ac}_on_time_completion_fraction"]=float(np.sum(samples[ac]<=deadline)/max(int(float(row[f"{ac}_at_risk_packets"])),1))
        per_seed.append(item)
    write(out/"per_seed.csv",per_seed)
    metrics=("target_offered_mbps","goodput_mbps","external_collision_rate","delivery_fraction","final_failure_fraction","unfinished_fraction",
             "vo_delivery_fraction","vi_delivery_fraction","be_delivery_fraction","bk_delivery_fraction",
             "vo_on_time_completion_fraction","vi_on_time_completion_fraction","vo_delay_p95_ms","vi_delay_p95_ms","be_delay_p95_ms","bk_delay_p95_ms")
    summary=[]
    for scenario in sorted({r["scenario"] for r in per_seed}):
        group=[r for r in per_seed if r["scenario"]==scenario]; item={"scenario":scenario,"seeds":len(group)}
        for metric in metrics:
            values=[float(r[metric]) for r in group]; item[metric+"_mean"]=np.mean(values); item[metric+"_ci95"]=ci95(values)
        summary.append(item)
    write(out/"summary.csv",summary)
    (out/"provenance.json").write_text(json.dumps({"run":str(run),"accounting_version":3,"seeds":[631,632,633],
        "aggregation":"each ratio is computed per seed, followed by arithmetic mean and two-sided t CI (df=2)",
        "on_time_note":"successful ACK delay <= 20/50 ms divided by measurement-window at-risk packets; end-pending packets are conservatively not on-time, so this is a boundary-sensitive lower bound pending Stage-F deadline ledger",
        "delay":"generation immediately before MAC Send to successful ACK; successful packets only"},indent=2),encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Times","DejaVu Serif"],"font.size":9,"pdf.fonttype":42,"savefig.dpi":300})
    labels=[x["scenario"].split("_")[0] for x in summary]; x=np.arange(len(summary)); width=.19
    fig,ax=plt.subplots(figsize=(8,4))
    for i,ac in enumerate(ACS):
        values=[r[f"{ac}_delivery_fraction_mean"] for r in summary]; errors=[r[f"{ac}_delivery_fraction_ci95"] for r in summary]
        ax.bar(x+(i-1.5)*width,values,width,yerr=errors,capsize=2,label=ac.upper())
    ax.set(xticks=x,xticklabels=labels,ylabel="Delivered / at-risk packets",ylim=(0,1.08)); ax.grid(axis="y",alpha=.2); ax.legend(ncol=4)
    fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(figs/f"per_ac_delivery.{ext}")
    plt.close(fig)

    for ac,deadline in (("vo",20.0),("vi",50.0)):
        fig,ax=plt.subplots(figsize=(6.4,4.2))
        all_data={}
        for scenario in sorted({r["scenario"] for r in rows}):
            all_data[scenario]=[np.load(run/f"{scenario}_seed{seed}"/"delay_samples.npz")[ac] for seed in (631,632,633)]
        positive=np.concatenate([v for values in all_data.values() for v in values if len(v)])
        grid=np.unique(np.geomspace(max(.05,float(positive.min())),float(positive.max()),600))
        for scenario,values in all_data.items():
            curves=[empirical_ccdf(v,grid) for v in values]; curve=np.mean(curves,axis=0); valid=curve>0
            ax.step(grid[valid],curve[valid],where="post",label=scenario.split("_")[0])
        ax.axvline(deadline,color="#555",ls=":",lw=1,label=f"{deadline:g} ms target")
        ax.set(xscale="log",yscale="log",xlabel="Packet delay (ms)",ylabel="P(Delay > x)",title=f"Default EDCA {ac.upper()} delay CCDF")
        ax.grid(alpha=.2,which="both"); ax.legend(ncol=2); fig.tight_layout()
        for ext in ("png","pdf"): fig.savefig(figs/f"{ac}_delay_ccdf.{ext}")
        plt.close(fig)

    periods=read(run/"periods.csv"); dynamic=[r for r in periods if r["scenario"]=="S5_dynamic_bursty"]
    fig,ax=plt.subplots(figsize=(7.2,4))
    for seed in (631,632,633):
        group=sorted((r for r in dynamic if int(r["eval_seed"])==seed),key=lambda r:float(r["time_s"]))
        ax.plot([float(r["time_s"]) for r in group],[float(r["offered_packets"]) for r in group],alpha=.65,label=f"seed {seed}")
    ax.set(xlabel="Measurement time (s)",ylabel="Generated packets per 1 s",title="S5 aggregate offered traffic (staggered STA phases)"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(figs/f"s5_network_offered_packets_time_series.{ext}")
    plt.close(fig)

    validation=read(ROOT/"results/raw/dynamic_traffic_validation_20261020/half_period_rates.csv")
    fig,ax=plt.subplots(figsize=(7.2,4))
    for seed in (621,622,623):
        group=sorted((r for r in validation if int(r["seed"])==seed),key=lambda r:int(r["half_period"]))
        times=[(int(r["half_period"])+.5)*2.5 for r in group]
        ax.step(times,[float(r["offered_mbps_sta1"]) for r in group],where="mid",alpha=.75,label=f"seed {seed}")
    ax.set(xlabel="Measurement time (s)",ylabel="STA 1 offered load (Mbit/s)",title="S5 per-STA two-state offered traffic"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(figs/f"s5_sta1_two_state_offered_load.{ext}")
    plt.close(fig)
    print(out); print(figs)


if __name__=="__main__": main()
