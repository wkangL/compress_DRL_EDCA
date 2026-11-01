#!/usr/bin/env python3
"""Generic final-checkpoint analysis for one-YAML MAC-CMDP experiments."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ACS = ("vo", "vi", "be", "bk")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_rows(root: Path, train_seeds: list[int], eval_seeds: list[int], held_out_dir: str = "held_out") -> list[dict]:
    rows = []
    for train_seed in train_seeds:
        for eval_seed in eval_seeds:
            for policy in ("default", "final"):
                path = root / f"seed_{train_seed}" / held_out_dir / f"seed_{eval_seed}" / policy / "evaluation_summary.json"
                data = json.loads(path.read_text(encoding="utf-8"))
                row = {"train_seed": train_seed, "eval_seed": eval_seed, "policy": policy,
                       "reward": data["reward_mean"], "cmdp_utility": data.get("cmdp_utility_mean"),
                       "vo_timely": data.get("vo_on_time_delivery_rate"), "vi_timely": data.get("vi_on_time_delivery_rate"),
                       "vo_satisfaction": data.get("vo_demand_satisfaction"), "vi_satisfaction": data.get("vi_demand_satisfaction"),
                       "be_satisfaction": data.get("be_demand_satisfaction"), "bk_satisfaction": data.get("bk_demand_satisfaction"),
                       "jain": data.get("jain_demand_satisfaction")}
                for ac in ACS:
                    row[f"{ac}_throughput_mbps"] = sum(float(data.get(f"{d}_{ac}_throughput_mbps") or 0) for d in ("ul", "dl"))
                    for d in ("ul", "dl"):
                        row[f"{d}_{ac}_throughput_mbps"] = data.get(f"{d}_{ac}_throughput_mbps")
                        for suffix in ("delay_p95_ms", "final_drop_rate", "retransmission_rate", "risk_packets", "successful_samples", "pending_end"):
                            row[f"{d}_{ac}_{suffix}"] = data.get(f"{d}_{ac}_{suffix}")
                rows.append(row)
    return rows


def unique_default(rows):
    return list({int(r["eval_seed"]): r for r in rows if r["policy"] == "default"}.values())


def numeric(rows, key):
    return np.asarray([float(r[key]) for r in rows if r.get(key) not in (None, "")], dtype=float)


def aggregate(rows):
    output=[]; drows=unique_default(rows); frows=[r for r in rows if r["policy"]=="final"]
    keys=("reward","cmdp_utility","vo_timely","vi_timely","vo_satisfaction","vi_satisfaction","be_satisfaction",
          "bk_satisfaction","jain","vo_throughput_mbps","vi_throughput_mbps","be_throughput_mbps","bk_throughput_mbps")
    for key in keys:
        d,f=numeric(drows,key),numeric(frows,key)
        output.append({"metric":key,"default_mean":d.mean(),"default_std":d.std(ddof=1) if len(d)>1 else 0,
                       "default_n":len(d),"final_mean":f.mean(),"final_std":f.std(ddof=1) if len(f)>1 else 0,
                       "final_n":len(f),"relative_change_percent":((f.mean()/d.mean()-1)*100 if d.mean()!=0 else None)})
    return output


def plot_diagnostics(root, figures, rows, train_seeds):
    figures.mkdir(parents=True,exist_ok=True); d=unique_default(rows); f=[r for r in rows if r["policy"]=="final"]
    fig,ax=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
    keys=("vo_timely","vi_timely","jain"); labels=("VO timely","VI timely","Jain"); x=np.arange(3)
    for off,name,data,color in ((-.18,"Default",d,"#4c78a8"),(.18,"Final",f,"#e45756")):
        ax[0].bar(x+off,[numeric(data,k).mean() for k in keys],.36,label=name,color=color)
    ax[0].set_xticks(x,labels); ax[0].set_ylim(0,1); ax[0].legend(); ax[0].set_title("Final policy versus Default")
    for seed in train_seeds:
        steps=read_csv(root/f"seed_{seed}"/"training_steps.csv"); losses=read_csv(root/f"seed_{seed}"/"losses.csv")
        reward=np.asarray([float(r["reward"]) for r in steps]); n=min(20,len(reward))
        ax[1].plot(np.arange(n,len(reward)+1),np.convolve(reward,np.ones(n)/n,"valid"),label=f"seed {seed}")
        loss=np.asarray([float(r["td_loss"]) for r in losses]); n=min(20,len(loss))
        ax[2].plot(np.arange(n,len(loss)+1),np.convolve(loss,np.ones(n)/n,"valid"),label=f"seed {seed}")
    ax[1].set(title="Training reward (20-step mean)",xlabel="environment step"); ax[2].set(title="Q loss (20-update mean)",xlabel="gradient update")
    ax[1].legend(); ax[2].legend()
    fig.savefig(figures/"final_comparison_and_training_diagnostics.png",dpi=220); plt.close(fig)


def ccdf_curve(folder: Path, ac: str, direction: str|None, grid: np.ndarray):
    data=read_csv(folder/"mac_service_delay_ccdf.csv")
    summary=json.loads((folder/"evaluation_summary.json").read_text(encoding="utf-8")); out=np.zeros_like(grid); total=0
    for d in ((direction,) if direction else ("UL","DL")):
        part=[r for r in data if r["scope"]=="BSS" and r["ac"]==ac and r["direction"]==d]
        n=int(summary[f"{d.lower()}_{ac.lower()}_successful_samples"])
        if not part or not n: continue
        x=np.asarray([float(r["delay_ms"]) for r in part]); y=np.asarray([float(r["ccdf"]) for r in part]); order=np.argsort(x)
        out += n*np.interp(grid,x[order],y[order],left=1,right=0); total += n
    return out/total if total else np.full_like(grid,np.nan)


def plot_ccdf(root, figures, train_seeds, eval_seeds, held_out_dir="held_out"):
    grid=np.linspace(0,500,2001); colors=["#222222","#1f77b4","#ff7f0e","#2ca02c","#9467bd"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for ax,ac in zip(axes,("VO","VI")):
        groups=[("Default",[ccdf_curve(root/f"seed_{train_seeds[0]}"/held_out_dir/f"seed_{e}"/"default",ac,None,grid) for e in eval_seeds])]
        groups += [(f"Final-{s}",[ccdf_curve(root/f"seed_{s}"/held_out_dir/f"seed_{e}"/"final",ac,None,grid) for e in eval_seeds]) for s in train_seeds]
        for color,(name,curves) in zip(colors,groups):
            arr=np.asarray(curves); mean=np.nanmean(arr,axis=0)*100
            ax.plot(grid,mean,lw=2.2,color=color,label=name)
            if len(arr)>1: ax.fill_between(grid,np.maximum(np.nanmin(arr,axis=0)*100,.05),np.maximum(np.nanmax(arr,axis=0)*100,.05),color=color,alpha=.07)
        ax.set(yscale="log",xlim=(0,500),ylim=(.05,100),xlabel="successful MPDU MAC service delay / ms",ylabel="tail probability / %",title=f"{ac} pooled UL+DL")
        ax.set_yticks([100,50,10,1,.1,.05],["100","50","10","1","0.1","0.05"]); ax.grid(True,which="both",alpha=.22)
    axes[0].legend(loc="lower left"); fig.savefig(figures/"final_mac_service_delay_ccdf_pooled.png",dpi=240); plt.close(fig)


def plot_comprehensive_metrics(figures: Path, rows: list[dict]) -> None:
    """Direction/AC figures requested for final-policy review."""
    drows, frows = unique_default(rows), [r for r in rows if r["policy"] == "final"]
    categories = [(d, ac) for ac in ACS for d in ("ul", "dl")]
    labels = [f"{d.upper()}-{ac.upper()}" for d, ac in categories]
    x = np.arange(len(categories)); width = .36

    def grouped_figure(suffix: str, ylabel: str, filename: str, title: str, percent: bool = False):
        fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
        for offset, name, source, color in ((-.18, "Default", drows, "#4c78a8"),
                                             (.18, "Final", frows, "#e45756")):
            means=[]; errors=[]
            for direction, ac in categories:
                values=numeric(source, f"{direction}_{ac}_{suffix}")
                means.append(float(values.mean()) if len(values) else np.nan)
                errors.append(float(values.std(ddof=1)) if len(values)>1 else 0.)
            scale=100. if percent else 1.
            ax.bar(x+offset, np.asarray(means)*scale, width, yerr=np.asarray(errors)*scale,
                   capsize=2.5, label=name, color=color, alpha=.9)
        ax.set_xticks(x,labels,rotation=35,ha="right"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(axis="y",alpha=.22); ax.legend()
        fig.savefig(figures/filename,dpi=240); plt.close(fig)

    grouped_figure("throughput_mbps", "successful MAC payload throughput / Mbps",
                   "throughput_by_ac_direction.png", "Successful throughput by AC and direction")
    grouped_figure("final_drop_rate", "final failure rate / %",
                   "final_drop_rate_by_ac_direction.png", "Final MAC failure rate by AC and direction", True)
    grouped_figure("retransmission_rate", "retransmissions / transmission attempts / %",
                   "retransmission_rate_collision_diagnostic.png",
                   "Retransmission rate (collision rate is not identifiable in summary traces)", True)

    fig, ax=plt.subplots(figsize=(6.8,4.8),constrained_layout=True)
    for offset,name,source,color in ((-.18,"Default",drows,"#4c78a8"),(.18,"Final",frows,"#e45756")):
        values=numeric(source,"jain")
        ax.bar([offset],[values.mean()],.32,yerr=[values.std(ddof=1) if len(values)>1 else 0],
               capsize=4,label=name,color=color)
    ax.set_xticks([0],["demand-normalized Jain"]); ax.set_ylim(0,1); ax.set_ylabel("Jain fairness")
    ax.set_title("Fairness across STAs with application demand"); ax.grid(axis="y",alpha=.22); ax.legend()
    fig.savefig(figures/"jain_demand_fairness.png",dpi=240); plt.close(fig)


def plot_ccdf_by_direction_all_ac(root: Path, figures: Path, train_seed: int, eval_seeds: list[int], held_out_dir="held_out") -> None:
    grid=np.linspace(0,500,2001)
    fig,axes=plt.subplots(4,2,figsize=(13,15),constrained_layout=True)
    for row,ac in enumerate(("VO","VI","BE","BK")):
        for col,direction in enumerate(("UL","DL")):
            ax=axes[row,col]
            plotted=False
            for name,policy,color in (("Default","default","#222222"),("Final","final","#d62728")):
                curves=[]
                for seed in eval_seeds:
                    curve=ccdf_curve(root/f"seed_{train_seed}"/held_out_dir/f"seed_{seed}"/policy,
                                     ac,direction,grid)
                    if not np.isnan(curve).all(): curves.append(curve)
                if not curves: continue
                arr=np.asarray(curves); mean=np.nanmean(arr,axis=0)*100
                ax.plot(grid,mean,lw=2.1,color=color,label=name); plotted=True
                if len(arr)>1:
                    ax.fill_between(grid,np.maximum(np.nanmin(arr,axis=0)*100,.1),
                                    np.maximum(np.nanmax(arr,axis=0)*100,.1),color=color,alpha=.08)
            if not plotted:
                ax.text(.5,.5,"No successful MPDU samples",ha="center",va="center",transform=ax.transAxes)
            ax.set(yscale="log",xlim=(0,500),ylim=(.1,100),title=f"{direction}-{ac}",
                   xlabel="successful MPDU MAC service delay / ms",ylabel="tail probability / %")
            ax.set_yticks([100,50,10,1,.1],["100","50","10","1","0.1"]); ax.grid(True,which="both",alpha=.22)
    axes[0,0].legend(loc="lower left")
    fig.suptitle("Final MAC service-delay CCDF by direction and AC")
    fig.savefig(figures/"mac_service_delay_ccdf_by_ac_direction.png",dpi=240); plt.close(fig)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--summary",type=Path,required=True); p.add_argument("--figures",type=Path,required=True)
    p.add_argument("--train-seeds",type=int,nargs="+",required=True); p.add_argument("--eval-seeds",type=int,nargs="+",required=True)
    p.add_argument("--held-out-dir",default="held_out"); a=p.parse_args()
    rows=load_rows(a.root,a.train_seeds,a.eval_seeds,a.held_out_dir); a.summary.mkdir(parents=True,exist_ok=True)
    write_csv(a.summary/"final_checkpoint_vs_default_per_run.csv",rows); write_csv(a.summary/"final_checkpoint_vs_default_aggregate.csv",aggregate(rows))
    plot_diagnostics(a.root,a.figures,rows,a.train_seeds); plot_ccdf(a.root,a.figures,a.train_seeds,a.eval_seeds,a.held_out_dir)
    plot_comprehensive_metrics(a.figures,rows)
    if len(a.train_seeds)==1:
        plot_ccdf_by_direction_all_ac(a.root,a.figures,a.train_seeds[0],a.eval_seeds,a.held_out_dir)
    print(json.dumps({"status":"complete","runs":len(rows),"summary":str(a.summary),"figures":str(a.figures)},indent=2))


if __name__=="__main__": main()
