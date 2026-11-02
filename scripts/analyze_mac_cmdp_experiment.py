#!/usr/bin/env python3
"""Paper-style final-checkpoint analysis for one-YAML MAC-CMDP experiments.

Delay CCDFs use measured successful-MPDU records and are rendered as empirical
step functions. Failed, expired, and unfinished MPDUs remain separate metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ACS = ("vo", "vi", "be", "bk")
CATEGORIES = (("ul", "vo"), ("dl", "vo"), ("ul", "vi"), ("dl", "vi"),
              ("ul", "be"), ("dl", "be"), ("ul", "bk"), ("dl", "bk"))
DEFAULT_COLOR = "#4C78A8"
FINAL_COLOR = "#E45756"
GRID_COLOR = "#D9D9D9"


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "legend.fontsize": 9, "axes.linewidth": .8,
        "lines.linewidth": 2.0, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight",
    })


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def evaluation_base(root: Path, train_seed: int, held_out_dir: str) -> Path:
    """Support both normal training roots and evaluation-only result roots."""
    normal=root/f"seed_{train_seed}"/held_out_dir
    direct=root/held_out_dir
    return normal if normal.is_dir() else direct


def load_rows(root: Path, train_seeds: list[int], eval_seeds: list[int], held_out_dir: str = "held_out") -> list[dict]:
    rows = []
    for train_seed in train_seeds:
        for eval_seed in eval_seeds:
            for policy in ("default", "final"):
                path = evaluation_base(root,train_seed,held_out_dir)/f"seed_{eval_seed}"/policy/"evaluation_summary.json"
                data = json.loads(path.read_text(encoding="utf-8"))
                row = {"train_seed": train_seed, "eval_seed": eval_seed, "policy": policy,
                       "reward": data["reward_mean"], "cmdp_utility": data.get("cmdp_utility_mean"),
                       "vo_timely": data.get("vo_on_time_delivery_rate"), "vi_timely": data.get("vi_on_time_delivery_rate"),
                       "vo_satisfaction": data.get("vo_demand_satisfaction"), "vi_satisfaction": data.get("vi_demand_satisfaction"),
                       "be_satisfaction": data.get("be_demand_satisfaction"), "bk_satisfaction": data.get("bk_demand_satisfaction"),
                       "jain": data.get("jain_demand_satisfaction")}
                for ac in ACS:
                    row[f"{ac}_throughput_mbps"] = sum(float(data.get(f"{d}_{ac}_throughput_mbps") or 0) for d in ("ul", "dl"))
                    for direction in ("ul", "dl"):
                        row[f"{direction}_{ac}_throughput_mbps"] = data.get(f"{direction}_{ac}_throughput_mbps")
                        for suffix in ("delay_p95_ms", "final_drop_rate", "retransmission_rate", "risk_packets", "successful_samples", "pending_end"):
                            row[f"{direction}_{ac}_{suffix}"] = data.get(f"{direction}_{ac}_{suffix}")
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


def mean_value(rows: list[dict], key: str) -> float:
    values = numeric(rows, key)
    return float(values.mean()) if len(values) else np.nan


def polish_axis(ax, grid_axis="y") -> None:
    ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=.7, alpha=.65)
    ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)


def add_value(ax, x: float, y: float, fmt: str = ".3g") -> None:
    if np.isfinite(y):
        ax.annotate(format(y,fmt),(x,y),xytext=(0,5),textcoords="offset points",ha="center",fontsize=8.5)


def plot_diagnostics(root, figures, rows, train_seeds):
    figures.mkdir(parents=True,exist_ok=True); d=unique_default(rows); f=[r for r in rows if r["policy"]=="final"]
    fig,ax=plt.subplots(1,3,figsize=(13.2,3.8),constrained_layout=True)
    keys=("vo_timely","vi_timely","jain"); labels=("VO timely","VI timely","Jain"); x=np.arange(3)
    for off,name,data,color in ((-.18,"Default",d,DEFAULT_COLOR),(.18,"Final",f,FINAL_COLOR)):
        values=[mean_value(data,k) for k in keys]; ax[0].bar(x+off,values,.36,label=name,color=color)
        for xpos,value in zip(x+off,values): add_value(ax[0],xpos,value,".3f")
    ax[0].set_xticks(x,labels); ax[0].set_ylim(0,1.08); ax[0].legend(loc="upper center",ncol=2,frameon=False)
    ax[0].set_title("Final policy versus Default"); polish_axis(ax[0])
    for seed in train_seeds:
        steps=read_csv(root/f"seed_{seed}"/"training_steps.csv"); losses=read_csv(root/f"seed_{seed}"/"losses.csv")
        reward=np.asarray([float(r["reward"]) for r in steps]); n=min(20,len(reward))
        ax[1].plot(np.arange(n,len(reward)+1),np.convolve(reward,np.ones(n)/n,"valid"),label=f"seed {seed}")
        loss=np.asarray([float(r["td_loss"]) for r in losses]); n=min(20,len(loss))
        if n: ax[2].plot(np.arange(n,len(loss)+1),np.convolve(loss,np.ones(n)/n,"valid"),label=f"seed {seed}")
    ax[1].set(title="Training reward (20-step mean)",xlabel="Environment step")
    ax[2].set(title="Q loss (20-update mean)",xlabel="Gradient update")
    for axis in ax[1:]: polish_axis(axis); axis.legend(frameon=False)
    fig.savefig(figures/"final_comparison_and_training_diagnostics.png",dpi=300); plt.close(fig)


def ccdf_lower_percent(n: int) -> float | None:
    if n < 20: return None
    if n < 100: return 5.0
    if n < 1000: return 1.0
    return .1


def load_ccdf(folder: Path, ac: str, direction: str) -> tuple[np.ndarray,np.ndarray,int,float|None]:
    """Load one measured CCDF without interpolation or smoothing."""
    summary=json.loads((folder/"evaluation_summary.json").read_text(encoding="utf-8"))
    n=int(summary.get(f"{direction.lower()}_{ac.lower()}_successful_samples") or 0); p99=None
    summary_path=folder/"mac_service_delay_summary.csv"
    if summary_path.is_file():
        matches=[r for r in read_csv(summary_path) if r["scope"]=="BSS" and r["ac"]==ac and r["direction"]==direction]
        if matches and matches[0].get("p99_ms") not in (None,""): p99=float(matches[0]["p99_ms"])
    if n==0: return np.asarray([]),np.asarray([]),0,p99
    part=[r for r in read_csv(folder/"mac_service_delay_ccdf.csv")
          if r["scope"]=="BSS" and r["ac"]==ac and r["direction"]==direction]
    x=np.asarray([float(r["delay_ms"]) for r in part]); y=np.asarray([float(r["ccdf"])*100 for r in part])
    order=np.argsort(x,kind="stable")
    if p99 is None and len(x):
        eligible=x[y<=1.0]; p99=float(eligible[0] if len(eligible) else x[-1])
    return x[order],y[order],n,p99


def policy_folders(root: Path, train_seeds: list[int], eval_seeds: list[int], held_out_dir: str, policy: str) -> list[Path]:
    if policy=="default":
        return [evaluation_base(root,train_seeds[0],held_out_dir)/f"seed_{seed}"/"default" for seed in eval_seeds]
    return [evaluation_base(root,train_seed,held_out_dir)/f"seed_{eval_seed}"/"final"
            for train_seed in train_seeds for eval_seed in eval_seeds]


def combine_empirical_ccdfs(folders: list[Path], ac: str, direction: str) -> tuple[np.ndarray,np.ndarray,int,float|None]:
    """Pool measured step distributions using exact sample-count weights."""
    curves=[]; total=0; p99s=[]
    for folder in folders:
        x,y,n,p99=load_ccdf(folder,ac,direction)
        if n:
            curves.append((x,y,n)); total+=n
            if p99 is not None: p99s.append(p99)
    if not curves: return np.asarray([]),np.asarray([]),0,None
    grid=np.unique(np.concatenate([curve[0] for curve in curves])); weighted=np.zeros(len(grid))
    for x,y,n in curves:
        indices=np.searchsorted(x,grid,side="right")-1
        values=np.where(indices<0,100.0,y[np.maximum(indices,0)])
        weighted+=n*values
    return grid,weighted/total,total,(max(p99s) if p99s else None)


def plot_ccdf_panel(root: Path, figures: Path, train_seeds: list[int], eval_seeds: list[int], held_out_dir: str,
                    categories: tuple[tuple[str,str],...], filename: str, title: str) -> None:
    fig,axes=plt.subplots(2,2,figsize=(11.2,7.4),constrained_layout=True)
    for ax,(direction,ac) in zip(axes.flat,categories):
        default=combine_empirical_ccdfs(policy_folders(root,train_seeds,eval_seeds,held_out_dir,"default"),ac.upper(),direction.upper())
        final=combine_empirical_ccdfs(policy_folders(root,train_seeds,eval_seeds,held_out_dir,"final"),ac.upper(),direction.upper())
        nx,ny,nd,p99d=default; fx,fy,nf,p99f=final; lower=ccdf_lower_percent(min(nd,nf))
        if nd==0 or nf==0:
            messages=[]
            if nd==0: messages.append("Default: No successful MPDU samples")
            if nf==0: messages.append("Final: No successful MPDU samples")
            ax.text(.5,.53,"\n".join(messages),ha="center",va="center",transform=ax.transAxes)
        elif lower is None:
            ax.text(.5,.54,f"Insufficient successful MPDU samples\nDefault N={nd:,}; Final N={nf:,}\n(CCDF requires N ≥ 20)",
                    ha="center",va="center",transform=ax.transAxes)
        else:
            ax.step(nx,ny,where="post",color=DEFAULT_COLOR,label=f"Default (N={nd:,})")
            ax.step(fx,fy,where="post",color=FINAL_COLOR,label=f"Final (N={nf:,})")
            xmax=max(v for v in (p99d,p99f) if v is not None)*1.05
            ax.set_xlim(0,max(xmax,1e-6)); ax.set_ylim(lower,100); ax.set_yscale("log")
            ticks=[tick for tick in (100,50,10,5,1,.1) if tick>=lower]
            ax.set_yticks(ticks,[f"{tick:g}" for tick in ticks]); ax.legend(loc="upper right",frameon=False)
        ax.set_title(f"{ac.upper()}-{direction.upper()}"); ax.set_xlabel("MAC service delay (ms)"); ax.set_ylabel("Tail probability (%)")
        ax.grid(True,which="both",color=GRID_COLOR,linewidth=.7,alpha=.65); ax.spines[["top","right"]].set_visible(False)
        if lower is None: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title,fontsize=13); fig.savefig(figures/filename,dpi=300); plt.close(fig)


def dumbbell(figures: Path, drows: list[dict], frows: list[dict], categories: tuple[tuple[str,str],...],
             suffix: str, ylabel: str, filename: str, title: str, scale: float=1.0) -> None:
    labels=[f"{ac.upper()}-{direction.upper()}" for direction,ac in categories]
    default=np.asarray([mean_value(drows,f"{direction}_{ac}_{suffix}") for direction,ac in categories])*scale
    final=np.asarray([mean_value(frows,f"{direction}_{ac}_{suffix}") for direction,ac in categories])*scale
    y=np.arange(len(categories))[::-1]; fig,ax=plt.subplots(figsize=(8.2,3.7),constrained_layout=True)
    for ypos,a,b in zip(y,default,final):
        if np.isfinite(a) and np.isfinite(b): ax.plot([a,b],[ypos,ypos],color="#A9A9A9",lw=2,zorder=1)
    ax.scatter(default,y,s=58,color=DEFAULT_COLOR,label="Default",zorder=3); ax.scatter(final,y,s=58,color=FINAL_COLOR,label="Final",zorder=3)
    for ypos,a,b in zip(y,default,final):
        if np.isfinite(a):
            ax.annotate(f"{a:.3g}",(a,ypos),xytext=(-8,7),textcoords="offset points",ha="right",va="bottom",fontsize=8,color=DEFAULT_COLOR)
        if np.isfinite(b):
            ax.annotate(f"{b:.3g}",(b,ypos),xytext=(8,-7),textcoords="offset points",ha="left",va="top",fontsize=8,color=FINAL_COLOR)
    ax.set_yticks(y,labels); ax.set_xlabel(ylabel); ax.set_title(title)
    ax.legend(loc="upper center",ncol=2,frameon=False); polish_axis(ax,"x")
    fig.savefig(figures/filename,dpi=300); plt.close(fig)


def relative_improvement(default: float, final: float, higher_is_better: bool) -> float:
    if not np.isfinite(default) or not np.isfinite(final) or default==0: return np.nan
    change=(final/default-1)*100
    return change if higher_is_better else -change


def plot_comprehensive_metrics(figures: Path, rows: list[dict]) -> None:
    drows,frows=unique_default(rows),[r for r in rows if r["policy"]=="final"]
    for cats,name,label in ((CATEGORIES[:4],"high_priority","High-priority (VO/VI)"),(CATEGORIES[4:],"low_priority","Low-priority (BE/BK)")):
        dumbbell(figures,drows,frows,cats,"throughput_mbps","Successful MAC payload throughput (Mbps)",f"throughput_{name}.png",f"{label} throughput")
        dumbbell(figures,drows,frows,cats,"final_drop_rate","Final MAC failure rate (%)",f"final_failure_rate_{name}.png",f"{label} final MAC failure rate",100)
        dumbbell(figures,drows,frows,cats,"retransmission_rate","Retransmissions / transmission attempts (%)",f"retransmission_rate_{name}.png","MAC retransmission rate by AC and direction",100)

    default=mean_value(drows,"jain"); final=mean_value(frows,"jain"); change=relative_improvement(default,final,True)
    fig,ax=plt.subplots(figsize=(5.8,2.25),constrained_layout=True)
    ax.plot([default,final],[0,0],color="#A9A9A9",lw=2); ax.scatter([default],[0],s=75,color=DEFAULT_COLOR,label="Default",zorder=3)
    ax.scatter([final],[0],s=75,color=FINAL_COLOR,label="Final",zorder=3)
    ax.annotate(f"Default {default:.3f}",(default,0),xytext=(0,12),textcoords="offset points",ha="center",color=DEFAULT_COLOR)
    ax.annotate(f"Final {final:.3f}\n{change:+.1f}%",(final,0),xytext=(0,-30),textcoords="offset points",ha="center",color=FINAL_COLOR)
    ax.set_yticks([]); ax.set_xlabel("Demand-normalized Jain fairness"); ax.set_title("Fairness across STAs with application demand")
    ax.set_xlim(max(0,min(default,final)-.12),min(1,max(default,final)+.12)); ax.legend(loc="upper center",ncol=2,frameon=False)
    polish_axis(ax,"x"); fig.savefig(figures/"jain_demand_fairness.png",dpi=300); plt.close(fig)

    specs=(("throughput_mbps","Throughput",True),("final_drop_rate","Final failure",False),
           ("retransmission_rate","Retransmission",False),("delay_p95_ms","p95 delay",False))
    matrix=np.full((4,8),np.nan)
    for i,(suffix,_,higher) in enumerate(specs):
        for j,(direction,ac) in enumerate(CATEGORIES):
            matrix[i,j]=relative_improvement(mean_value(drows,f"{direction}_{ac}_{suffix}"),mean_value(frows,f"{direction}_{ac}_{suffix}"),higher)
    finite=np.abs(matrix[np.isfinite(matrix)]); limit=max(10,float(np.percentile(finite,95))) if len(finite) else 10
    fig,(ax,far)=plt.subplots(2,1,figsize=(11.2,5.7),height_ratios=(4,1),constrained_layout=True)
    image=ax.imshow(matrix,aspect="auto",cmap="RdYlGn",vmin=-limit,vmax=limit)
    ax.set_xticks(range(8),[f"{ac.upper()}-{direction.upper()}" for direction,ac in CATEGORIES]); ax.set_yticks(range(4),[s[1] for s in specs])
    ax.set_title("Default → Final improvement (%)")
    for i in range(4):
        for j in range(8):
            value=matrix[i,j]; ax.text(j,i,"N/A" if not np.isfinite(value) else f"{value:+.1f}%",ha="center",va="center",fontsize=8,
                    color="white" if np.isfinite(value) and abs(value)>.55*limit else "#222222")
    colorbar=fig.colorbar(image,ax=ax,pad=.015); colorbar.set_label("Improvement (%) — positive is better")
    far.barh([0],[change],color="#59A14F" if change>=0 else "#E15759",height=.45); far.axvline(0,color="#666666",lw=.8)
    far.text(change,0,f"  Jain fairness {change:+.1f}%",va="center",ha="left" if change>=0 else "right")
    far.set_yticks([]); far.set_xlabel("Improvement (%)"); polish_axis(far,"x")
    fig.savefig(figures/"improvement_percentage_summary.png",dpi=300); plt.close(fig)


def plot_split_profile_distribution(root: Path, figures: Path, train_seeds: list[int],
                                    eval_seeds: list[int], held_out_dir: str) -> None:
    """Plot AP/STA profile selections when the evaluated policy has split branches."""
    totals={"ap_profile":{},"sta_profile":{}}
    for folder in policy_folders(root,train_seeds,eval_seeds,held_out_dir,"final"):
        summary=json.loads((folder/"evaluation_summary.json").read_text(encoding="utf-8"))
        for branch in summary.get("action_distribution",[]):
            name=branch.get("branch")
            if name not in totals: continue
            for index,count in branch.get("counts",{}).items():
                totals[name][int(index)]=totals[name].get(int(index),0)+int(count)
    if not any(totals.values()): return
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.6),constrained_layout=True)
    for ax,name,title,color in ((axes[0],"ap_profile","AP Profile",DEFAULT_COLOR),
                                (axes[1],"sta_profile","STA Profile",FINAL_COLOR)):
        keys=sorted(totals[name]); values=np.asarray([totals[name][key] for key in keys],dtype=float)
        fractions=values/values.sum()*100 if values.sum() else values
        bars=ax.bar([f"P{key}" for key in keys],fractions,color=color,width=.65)
        for bar,value,count in zip(bars,fractions,values.astype(int)):
            ax.annotate(f"{value:.1f}%\n(n={count})",(bar.get_x()+bar.get_width()/2,bar.get_height()),
                        xytext=(0,4),textcoords="offset points",ha="center",va="bottom",fontsize=8.5)
        ax.set_title(title); ax.set_ylabel("Selection share (%)"); ax.set_ylim(0,max(105,float(fractions.max())*1.22 if len(fractions) else 1))
        polish_axis(ax)
    fig.suptitle("Final greedy policy: separate AP and STA Profile selections",fontsize=12)
    fig.savefig(figures/"ap_sta_profile_action_distribution.png",dpi=300); plt.close(fig)


def remove_superseded_figures(figures: Path) -> None:
    for name in ("throughput_by_ac_direction.png","final_drop_rate_by_ac_direction.png","retransmission_rate_collision_diagnostic.png",
                 "mac_service_delay_ccdf_by_ac_direction.png","final_mac_service_delay_ccdf_pooled.png"):
        path=figures/name
        if path.is_file(): path.unlink()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--summary",type=Path,required=True); p.add_argument("--figures",type=Path,required=True)
    p.add_argument("--train-seeds",type=int,nargs="+",required=True); p.add_argument("--eval-seeds",type=int,nargs="+",required=True); p.add_argument("--held-out-dir",default="held_out")
    a=p.parse_args(); configure_style(); rows=load_rows(a.root,a.train_seeds,a.eval_seeds,a.held_out_dir)
    a.summary.mkdir(parents=True,exist_ok=True); a.figures.mkdir(parents=True,exist_ok=True); remove_superseded_figures(a.figures)
    write_csv(a.summary/"final_checkpoint_vs_default_per_run.csv",rows); write_csv(a.summary/"final_checkpoint_vs_default_aggregate.csv",aggregate(rows))
    if all((a.root/f"seed_{seed}"/"training_steps.csv").is_file() and
           (a.root/f"seed_{seed}"/"losses.csv").is_file() for seed in a.train_seeds):
        plot_diagnostics(a.root,a.figures,rows,a.train_seeds)
    elif (a.figures/"final_comparison_and_training_diagnostics.png").is_file():
        (a.figures/"final_comparison_and_training_diagnostics.png").unlink()
    plot_ccdf_panel(a.root,a.figures,a.train_seeds,a.eval_seeds,a.held_out_dir,CATEGORIES[:4],"mac_service_delay_ccdf_high_priority.png","High-priority MAC service-delay CCDF")
    plot_ccdf_panel(a.root,a.figures,a.train_seeds,a.eval_seeds,a.held_out_dir,CATEGORIES[4:],"mac_service_delay_ccdf_low_priority.png","Low-priority MAC service-delay CCDF")
    plot_comprehensive_metrics(a.figures,rows)
    plot_split_profile_distribution(a.root,a.figures,a.train_seeds,a.eval_seeds,a.held_out_dir)
    print(json.dumps({"status":"complete","runs":len(rows),"summary":str(a.summary),"figures":str(a.figures)},indent=2))


if __name__=="__main__": main()
