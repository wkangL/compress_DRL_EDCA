"""Report the controlled Reward-v1/v2/v2+service quick ablation."""
from __future__ import annotations
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT

VARIANTS=("baseline","reward_v1","reward_v2","reward_v2_service")
LABELS={"baseline":"Default EDCA","reward_v1":"Reward-v1",
        "reward_v2":"Reward-v2","reward_v2_service":"Reward-v2 + service"}
COLORS={"baseline":"#333333","reward_v1":"#0072B2","reward_v2":"#D55E00","reward_v2_service":"#009E73"}
METRICS=("goodput_mbps","collision_rate","drop_rate","vo_on_time_delivery_rate",
         "vi_on_time_delivery_rate","vo_delay_p95_ms","vi_delay_p95_ms","be_service","bk_service")
LOWER={"collision_rate","drop_rate","vo_delay_p95_ms","vi_delay_p95_ms"}


def read_csv(path):
    with path.open(encoding="utf-8") as stream: return list(csv.DictReader(stream))


def numeric_rows(path,variant):
    rows=[]
    for row in read_csv(path):
        item={"variant":variant,"scenario":row["scenario"]}
        for key in METRICS[:-2]: item[key]=float(row[key])
        for ac in ("be","bk"):
            offered=float(row[f"{ac}_offered_mbps"]); goodput=float(row[f"{ac}_goodput_mbps"])
            item[f"{ac}_service"]=min(goodput/max(offered,1e-12),1.) if offered>0 else 1.
        rows.append(item)
    return rows


def delay_path(eval_dir,row,ac):
    return eval_dir/f"{row['scenario']}_seed{row['eval_seed']}"/"delay_samples.npz",ac


def validate_configs(eval_dirs):
    reference=None
    for path in eval_dirs.values():
        config=json.loads((path/"config.json").read_text(encoding="utf-8"))
        comparable={k:config[k] for k in ("network","queues","timing","traffic","evaluation","model","training","action","state")}
        comparable["training"]={k:v for k,v in comparable["training"].items() if k not in ()}
        if reference is not None and comparable!=reference: raise ValueError("Reward ablation changed a controlled variable")
        reference=comparable


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v1",type=Path,required=True); p.add_argument("--v2",type=Path,required=True)
    p.add_argument("--v2-service",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(); roots={"reward_v1":args.v1,"reward_v2":args.v2,"reward_v2_service":args.v2_service}
    eval_dirs={k:v/"monitoring/stage_006/eval" for k,v in roots.items()}
    eval_dirs["baseline"]=args.v2/"monitoring/default/eval"; validate_configs(eval_dirs)
    out=args.output; figures=ROOT/"results/figures"/out.name; out.mkdir(parents=True,exist_ok=True); figures.mkdir(parents=True,exist_ok=True)
    all_rows=[]
    for variant,path in eval_dirs.items(): all_rows.extend(numeric_rows(path/"evaluation.csv",variant))
    lookup={(r["variant"],r["scenario"]):r for r in all_rows}; scenarios=sorted({r["scenario"] for r in all_rows})
    records=[]
    for row in all_rows:
        baseline=lookup["baseline",row["scenario"]]; v1=lookup["reward_v1",row["scenario"]]
        record=dict(row)
        for metric in METRICS:
            for name,reference in (("baseline",baseline),("v1",v1)):
                denominator=abs(reference[metric])
                change=None if denominator==0 else 100*(row[metric]-reference[metric])/denominator
                record[f"{metric}_change_vs_{name}_pct"]=change
                record[f"{metric}_improvement_vs_{name}_pct"]=(None if change is None else -change if metric in LOWER else change)
        records.append(record)
    fields=list(records[0])
    with (out/"reward_ablation_final.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(records)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,"savefig.dpi":300})
    # Successful-packet delay CCDFs, as required for all delay graphics.
    for ac in ("vo","vi"):
        ncols=2; nrows=int(np.ceil(len(scenarios)/ncols)); fig,axes=plt.subplots(nrows,ncols,figsize=(10,3.4*nrows),squeeze=False)
        for index,scenario in enumerate(scenarios):
            ax=axes.flat[index]
            for variant in VARIANTS:
                eval_rows=read_csv(eval_dirs[variant]/"evaluation.csv"); row=next(r for r in eval_rows if r["scenario"]==scenario)
                npz,key=delay_path(eval_dirs[variant],row,ac)
                with np.load(npz) as data: values=np.sort(data[key])
                if len(values):
                    probability=(len(values)-np.arange(len(values)))/len(values)
                    ax.step(values,probability,where="post",label=LABELS[variant],color=COLORS[variant],
                            linestyle="-" if variant=="baseline" else "--")
            ax.set_yscale("log"); ax.set_title(f"{scenario.split('_')[0]} / {ac.upper()}")
            ax.set_xlabel("MAC enqueue-to-ACK delay (ms)"); ax.set_ylabel("P(delay > x)"); ax.grid(alpha=.2,which="both")
        for index in range(len(scenarios),axes.size): axes.flat[index].set_visible(False)
        handles,labels=axes.flat[0].get_legend_handles_labels(); fig.legend(handles,labels,loc="lower center",ncol=4)
        fig.tight_layout(rect=(0,.08,1,1))
        for ext in ("png","pdf"): fig.savefig(figures/f"reward_delay_ccdf_{ac}.{ext}",bbox_inches="tight")
        plt.close(fig)

    # Keep TD-loss scales separate because reward magnitudes differ by definition.
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
    for ax,(variant,root) in zip(axes,roots.items()):
        rows=read_csv(root/"train/losses.csv"); x=np.array([int(r["update"]) for r in rows]); y=np.array([float(r["td_loss"]) for r in rows])
        window=min(25,len(y)); smooth=np.convolve(y,np.ones(window)/window,mode="valid")
        ax.plot(x,y,color=COLORS[variant],alpha=.18,lw=.6); ax.plot(x[window-1:],smooth,color=COLORS[variant],lw=1.4)
        ax.set_ylabel("TD loss"); ax.set_title(LABELS[variant]); ax.grid(alpha=.2)
    axes[-1].set_xlabel("Gradient update"); fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(figures/f"reward_td_loss.{ext}",bbox_inches="tight")
    plt.close(fig)

    # Independent VI QoS trend; each point is a separate greedy rollout.
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for variant,root in roots.items():
        paths=[]
        first=root.parent/"reward_v1_quick_seed11_20261021" if variant=="reward_v1" else None
        if first and (first/"monitoring/learning_curve.csv").exists(): paths.append(first/"monitoring/learning_curve.csv")
        paths.append(root/"monitoring/learning_curve.csv")
        rows=[]
        for path in paths: rows.extend(read_csv(path))
        stages=sorted({int(r["stage"]) for r in rows})
        ontime=[np.mean([float(r["vi_on_time_delivery_rate"]) for r in rows if int(r["stage"])==s]) for s in stages]
        p95=[np.mean([float(r["vi_delay_p95_ms"]) for r in rows if int(r["stage"])==s]) for s in stages]
        axes[0].plot(stages,ontime,marker="o",label=LABELS[variant],color=COLORS[variant]); axes[1].plot(stages,p95,marker="o",label=LABELS[variant],color=COLORS[variant])
    axes[0].set_ylabel("VI on-time delivery rate"); axes[1].set_ylabel("VI p95 delay (ms)")
    for ax in axes: ax.set_xlabel("Training stage"); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(figures/f"reward_independent_vi_trend.{ext}",bbox_inches="tight")
    plt.close(fig)

    provenance=dict(train_seed=11,evaluation_seed=301,stage=6,interaction_steps=120,gradient_updates=500,
        warning="Quick single-train-seed screening only; no convergence or significance claim",
        variants={k:str(v.resolve()) for k,v in roots.items()},baseline=str(eval_dirs["baseline"].resolve()),
        delay="successful-packet MAC enqueue-to-ACK CCDF",percent="100*(candidate-reference)/abs(reference)")
    (out/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    print(out)


if __name__=="__main__": main()
