"""Summarize Stage-E fixed actions without treating per-scenario optima as a deployable policy."""
from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT
from reporting.comparison import empirical_ccdf

METRICS=("goodput_mbps","external_collision_rate","delivery_fraction","final_failure_fraction","unfinished_fraction",
         "vo_on_time_completion_fraction","vi_on_time_completion_fraction","vo_delivery_fraction","vi_delivery_fraction",
         "be_delivery_fraction","bk_delivery_fraction","vo_delay_p95_ms","vi_delay_p95_ms","vo_delay_p99_ms","vi_delay_p99_ms")


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as stream: return list(csv.DictReader(stream))


def write(path: Path,rows: list[dict]) -> None:
    with path.open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def relative(new: float,base: float) -> float:
    return (new-base)/base if abs(base)>1e-12 else 0.0


def ci95(values: list[float]) -> float:
    factors={2:12.7062047364,3:4.3026527297}
    return factors.get(len(values),1.96)*np.std(values,ddof=1)/np.sqrt(len(values)) if len(values)>1 else 0.0


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--run",type=Path,default=ROOT/"results/raw/fixed_action_coarse_v2_schema3_20261020")
    parser.add_argument("--name",default="fixed_action_coarse_v2_schema3_20261020"); args=parser.parse_args()
    run=args.run.resolve()
    metadata=json.loads((run/"metadata.json").read_text(encoding="utf-8"))
    if metadata["state"]!="complete": raise ValueError("Fixed-action scan is incomplete")
    rows=read(run/"runs.csv"); out=ROOT/"results/summary"/args.name; figs=ROOT/"results/figures"/args.name
    out.mkdir(parents=True,exist_ok=True); figs.mkdir(parents=True,exist_ok=True)
    summary=[]
    for scenario in sorted({r["scenario"] for r in rows}):
        for action in metadata["actions"]:
            group=[r for r in rows if r["scenario"]==scenario and r["action"]==action]
            item={"scenario":scenario,"action":action,"seeds":len(group)}
            for metric in METRICS:
                values=[float(r[metric]) for r in group]; item[metric+"_mean"]=np.mean(values); item[metric+"_ci95"]=ci95(values); item[metric+"_min"]=min(values); item[metric+"_max"]=max(values)
            summary.append(item)
    write(out/"fixed_action_summary.csv",summary)
    comparisons=[]; selected={}
    for scenario in sorted({r["scenario"] for r in summary}):
        group=[r for r in summary if r["scenario"]==scenario]; base=next(r for r in group if r["action"]=="default_p0_beb")
        eligible=[]
        for row in group:
            item={"scenario":scenario,"action":row["action"]}
            for metric in METRICS: item[metric+"_relative_change"]=relative(row[metric+"_mean"],base[metric+"_mean"])
            item["vo_on_time_change_pp"]=100*(row["vo_on_time_completion_fraction_mean"]-base["vo_on_time_completion_fraction_mean"])
            item["vi_on_time_change_pp"]=100*(row["vi_on_time_completion_fraction_mean"]-base["vi_on_time_completion_fraction_mean"])
            item["be_delivery_change_pp"]=100*(row["be_delivery_fraction_mean"]-base["be_delivery_fraction_mean"])
            item["bk_delivery_change_pp"]=100*(row["bk_delivery_fraction_mean"]-base["bk_delivery_fraction_mean"])
            ok=(row["vo_on_time_completion_fraction_mean"]>=base["vo_on_time_completion_fraction_mean"]-.005 and
                row["vi_on_time_completion_fraction_mean"]>=base["vi_on_time_completion_fraction_mean"]-.005 and
                row["be_delivery_fraction_mean"]>=base["be_delivery_fraction_mean"]-.02 and
                row["bk_delivery_fraction_mean"]>=base["bk_delivery_fraction_mean"]-.02 and
                row["final_failure_fraction_mean"]<=base["final_failure_fraction_mean"]+.005)
            score=(item["goodput_mbps_relative_change"]-item["external_collision_rate_relative_change"]
                   -item["final_failure_fraction_relative_change"]-item["vo_delay_p95_ms_relative_change"]
                   -item["vi_delay_p95_ms_relative_change"]+.02*(item["vo_on_time_change_pp"]+item["vi_on_time_change_pp"]
                   +item["be_delivery_change_pp"]+item["bk_delivery_change_pp"]))
            item.update(screen_pass=ok,screen_score=score); comparisons.append(item)
            if ok: eligible.append((score,row["action"]))
        selected[scenario]=max(eligible)[1] if eligible else "default_p0_beb"
    write(out/"baseline_relative.csv",comparisons)
    (out/"refinement_candidates.json").write_text(json.dumps({"selected_by_constraint_screen":selected,
        "warning":"per-scenario candidates are diagnostic references, not a deployable adaptive policy",
        "constraints":"VO/VI on-time no worse than baseline by 0.5 pp; BE/BK delivery no worse by 2 pp; final failure no worse by 0.5 pp"},indent=2),encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Times","DejaVu Serif"],"font.size":8,"pdf.fonttype":42,"savefig.dpi":300})
    scenarios=sorted(selected); actions=list(metadata["actions"]); labels=[a.replace("_beb","").replace("growth_","g") for a in actions]
    for metric,title in (("delivery_fraction","Delivered / at-risk packets"),("external_collision_rate","External collision rate"),("vo_delay_p95_ms","VO p95 delay (ms)"),("vi_delay_p95_ms","VI p95 delay (ms)")):
        fig,axes=plt.subplots(1,len(scenarios),figsize=(3.1*len(scenarios),3.7),sharey=False)
        for ax,scenario in zip(np.atleast_1d(axes),scenarios):
            group=[r for r in summary if r["scenario"]==scenario]; mapping={r["action"]:r for r in group}
            values=[mapping[a][metric+"_mean"] for a in actions]
            colors=["#d62728" if a=="default_p0_beb" else "#1f77b4" for a in actions]
            ax.bar(np.arange(len(actions)),values,color=colors)
            ax.set(xticks=np.arange(len(actions)),xticklabels=labels,title=scenario.split("_")[0])
            ax.tick_params(axis="x",labelrotation=75); ax.grid(axis="y",alpha=.2)
        axes=np.atleast_1d(axes); axes[0].set_ylabel(title); fig.tight_layout()
        for ext in ("png","pdf"): fig.savefig(figs/f"{metric}.{ext}")
        plt.close(fig)

    for scenario,choice in selected.items():
        fig,axes=plt.subplots(1,2,figsize=(9,3.6))
        for ax,ac,deadline in zip(axes,("vo","vi"),(20,50)):
            for action in dict.fromkeys(("default_p0_beb",choice)):
                samples=[]
                for seed in metadata["seeds"]:
                    data=np.load(run/f"{scenario}__{action}__seed{seed}"/"delay_samples.npz")[ac]; samples.append(data)
                positive=np.concatenate([x for x in samples if len(x)]); grid=np.unique(np.geomspace(max(.05,float(positive.min())),float(positive.max()),500))
                curve=np.mean([empirical_ccdf(x,grid) for x in samples],axis=0); valid=curve>0
                ax.step(grid[valid],curve[valid],where="post",label=action)
            ax.axvline(deadline,color="#555",ls=":",lw=.8); ax.set(xscale="log",yscale="log",xlabel="Packet delay (ms)",ylabel="P(Delay > x)",title=ac.upper()); ax.grid(alpha=.2,which="both"); ax.legend(fontsize=6)
        fig.suptitle(f"{scenario}: Default vs constrained-screen candidate"); fig.tight_layout()
        for ext in ("png","pdf"): fig.savefig(figs/f"{scenario}_candidate_delay_ccdf.{ext}")
        plt.close(fig)
    print(out); print(figs); print(json.dumps(selected,indent=2))


if __name__=="__main__": main()
