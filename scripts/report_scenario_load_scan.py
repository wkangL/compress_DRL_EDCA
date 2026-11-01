"""Summarize baseline load-scan measurements and successful-packet VO/VI delay CCDFs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT
from reporting.comparison import empirical_ccdf


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as stream: return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def avg(rows: list[dict], key: str) -> float:
    return float(np.mean([float(r[key]) for r in rows]))


def panel_axes(plt, count: int):
    """Return a compact panel grid and hide unused axes."""
    cols=2 if count>1 else 1; rows=(count+cols-1)//cols
    fig,axes=plt.subplots(rows,cols,figsize=(5*cols,3.5*rows),squeeze=False,sharex=True)
    flat=list(axes.flat)
    for ax in flat[count:]: ax.set_visible(False)
    return fig,flat


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--name",default="scenario_load_scan_schema3_20261020")
    args=parser.parse_args(); run=args.run.resolve()
    metadata=json.loads((run/"metadata.json").read_text(encoding="utf-8"))
    if metadata["state"]!="complete" or metadata.get("accounting_version")!=3: raise ValueError("Incomplete schema-3 scan")
    rows=read_csv(run/"runs.csv"); scenarios=sorted({r["scenario"] for r in rows}); scales=sorted({float(r["scale"]) for r in rows})
    summary_dir=ROOT/"results/summary"/args.name; figure_dir=ROOT/"results/figures"/args.name
    summary_dir.mkdir(parents=True,exist_ok=True); figure_dir.mkdir(parents=True,exist_ok=True)
    metrics=("network_target_mbps","goodput_mbps","busy_fraction","delivery_fraction","final_failure_fraction",
             "unfinished_fraction","queue_rejection_fraction","external_collision_rate","ack_cts_timeout_ratio",
             "vo_delivery_fraction","vi_delivery_fraction","be_delivery_fraction","bk_delivery_fraction",
             "vo_delay_p95_ms","vo_delay_p99_ms","vi_delay_p95_ms","vi_delay_p99_ms",
             "be_delay_p95_ms","be_delay_p99_ms","bk_delay_p95_ms","bk_delay_p99_ms",
             "ul_vo_queue_p95","dl_vo_queue_p95","ul_vi_queue_p95","dl_vi_queue_p95")
    summary=[]
    for scenario in scenarios:
        for scale in scales:
            group=[r for r in rows if r["scenario"]==scenario and float(r["scale"])==scale]
            item={"scenario":scenario,"scale":scale,"seeds":len(group)}
            for metric in metrics:
                values=np.asarray([float(r[metric]) for r in group]); item[metric+"_mean"]=values.mean(); item[metric+"_min"]=values.min(); item[metric+"_max"]=values.max()
            qos=(item["vo_delay_p95_ms_mean"]<=20 and item["vi_delay_p95_ms_mean"]<=50)
            if item["delivery_fraction_mean"]>=.98 and qos: region="light_or_serviceable"
            elif item["delivery_fraction_mean"]<.80 or item["final_failure_fraction_mean"]+item["unfinished_fraction_mean"]>.20: region="sustained_overload"
            else: region="transition_or_competing_qos"
            item["operating_region_rule"]=region; summary.append(item)
    write_csv(summary_dir/"load_scan_summary.csv",summary)
    (summary_dir/"provenance.json").write_text(json.dumps({"run":str(run),"accounting_version":3,
        "aggregation":"mean and range over seeds at each scenario/scale; ratios computed per seed before aggregation",
        "region_rule":"descriptive only; serviceable requires delivery>=0.98 and VO/VI successful p95<=20/50ms",
        "delay":"MAC generation immediately before Send to successful ACK; successful packets only"},indent=2),encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Serif","font.size":9,"pdf.fonttype":42,"savefig.dpi":300})
    fig,axes=panel_axes(plt,len(scenarios))
    for ax,scenario in zip(axes,scenarios):
        current=[x for x in summary if x["scenario"]==scenario]
        x=[v["scale"] for v in current]
        ax.plot(x,[v["delivery_fraction_mean"] for v in current],"o-",label="Delivered")
        ax.plot(x,[v["final_failure_fraction_mean"] for v in current],"s-",label="Final failure")
        ax.plot(x,[v["unfinished_fraction_mean"] for v in current],"^-",label="Unfinished")
        ax.set(title=scenario.split("_")[0],xlabel="Historical load multiplier",ylabel="Packet fraction",ylim=(-.03,1.03)); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7); fig.suptitle("Default EDCA load regions (accounting schema 3)"); fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(figure_dir/f"load_delivery_regions.{ext}")
    plt.close(fig)

    fig,axes=panel_axes(plt,len(scenarios))
    for ax,scenario in zip(axes,scenarios):
        current=[x for x in summary if x["scenario"]==scenario]; x=[v["scale"] for v in current]
        ax.plot(x,[v["vo_delay_p95_ms_mean"] for v in current],"o-",label="VO p95")
        ax.plot(x,[v["vi_delay_p95_ms_mean"] for v in current],"s-",label="VI p95")
        ax.axhline(20,color="#1f77b4",ls=":",lw=.8); ax.axhline(50,color="#ff7f0e",ls=":",lw=.8)
        ax.set_yscale("log"); ax.set(title=scenario.split("_")[0],xlabel="Historical load multiplier",ylabel="Successful-packet delay (ms)"); ax.grid(alpha=.2,which="both")
    axes[0].legend(fontsize=7); fig.suptitle("Default EDCA successful-packet delay; absolute thresholds shown"); fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(figure_dir/f"load_delay_p95.{ext}")
    plt.close(fig)

    fig,axes=panel_axes(plt,len(scenarios))
    for ax,scenario in zip(axes,scenarios):
        current=[x for x in summary if x["scenario"]==scenario]; x=[v["scale"] for v in current]
        for ac,marker in zip(("vo","vi","be","bk"),("o","s","^","d")):
            ax.plot(x,[v[f"{ac}_delivery_fraction_mean"] for v in current],marker+"-",label=ac.upper())
        ax.set(title=scenario.split("_")[0],xlabel="Historical load multiplier",ylabel="Delivered / at-risk",ylim=(-.03,1.03)); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7); fig.suptitle("Per-AC service completion under Default EDCA"); fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(figure_dir/f"per_ac_delivery.{ext}")
    plt.close(fig)

    preferred=[x for x in scales if x in (0.40,0.70,1.00,1.15)]
    # Fine scans often use a different four-point grid; show all of it.
    selected=scales if len(scales)<=4 else preferred
    for scenario in scenarios:
        fig,axes=plt.subplots(1,2,figsize=(10,3.8))
        for ax,ac in zip(axes,("vo","vi")):
            all_samples={}
            for scale in selected:
                seed_samples=[]
                for row in [r for r in rows if r["scenario"]==scenario and float(r["scale"])==scale]:
                    case=run/f"{scenario}_scale{str(f'{scale:.2f}').replace('.', 'p')}_seed{row['seed']}"
                    data=np.load(case/"delay_samples.npz"); seed_samples.append(np.concatenate([data[f"ul_{ac}"],data[f"dl_{ac}"]]))
                all_samples[scale]=seed_samples
            positive=np.concatenate([v for values in all_samples.values() for v in values if len(v)])
            grid=np.unique(np.geomspace(max(0.05,float(np.min(positive))),max(0.051,float(np.max(positive))),500))
            for scale,values in all_samples.items():
                curves=[empirical_ccdf(v,grid) for v in values if len(v)]
                curve=np.mean(curves,axis=0); valid=curve>0
                ax.step(grid[valid],curve[valid],where="post",label=f"x{scale:.2f}")
            ax.axvline(20 if ac=="vo" else 50,color="#555555",ls=":",lw=.8,label="QoS target")
            ax.set(xscale="log",yscale="log",xlabel="Packet delay (ms)",ylabel="P(Delay > x)",title=ac.upper()); ax.grid(alpha=.2,which="both"); ax.legend(fontsize=7)
        fig.suptitle(f"{scenario}: successful-packet delay CCDF; equal seed weighting"); fig.tight_layout()
        for ext in ("png","pdf"): fig.savefig(figure_dir/f"{scenario}_vo_vi_delay_ccdf.{ext}")
        plt.close(fig)
    print(summary_dir); print(figure_dir)


if __name__=="__main__": main()
