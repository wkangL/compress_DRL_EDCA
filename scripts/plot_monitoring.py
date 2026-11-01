"""Independent greedy evaluation over training, with matched baseline rollouts."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT


def plot_monitoring(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (root/"monitoring/learning_curve.csv").open(encoding="utf-8") as stream: rows=list(csv.DictReader(stream))
    with (root/"monitoring/default/eval/evaluation.csv").open(encoding="utf-8") as stream: baseline=list(csv.DictReader(stream))
    output=ROOT/"results/figures"/root.name; output.mkdir(parents=True,exist_ok=True)
    scenarios=sorted({r["scenario"] for r in rows}); stages=sorted({int(r["stage"]) for r in rows})
    metrics={"reward":"Independent evaluation reward","goodput_mbps":"Goodput (Mbps)",
             "collision_rate":"Collision rate","drop_rate":"Drop rate","jain_fairness":"Jain fairness"}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,"savefig.dpi":300})
    for metric,label in metrics.items():
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        for ax,scenario in zip(axes.flat,scenarios):
            subset=[r for r in rows if r["scenario"]==scenario]; mean=[]; low=[]; high=[]
            for stage in stages:
                values=[float(r[metric]) for r in subset if int(r["stage"])==stage]
                mean.append(np.mean(values)); low.append(min(values)); high.append(max(values))
            ax.plot(stages,mean,"o-",color="#E69F00",label="Compressed Async + Age")
            ax.fill_between(stages,low,high,color="#E69F00",alpha=.15,label="Evaluation-seed range")
            ax.axhline(np.mean([float(r[metric]) for r in baseline if r["scenario"]==scenario]),
                color="#444444",linestyle="--",label="Default EDCA")
            ax.set_title(scenario.split('_')[0]); ax.set_xlabel("Completed training stage"); ax.set_ylabel(label); ax.grid(alpha=.2)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="lower center",ncol=3,fontsize=8)
        fig.suptitle("Greedy held-out monitoring; one training seed, no convergence claim",fontsize=10)
        fig.tight_layout(rect=(0,.06,1,.95))
        fig.savefig(output/f"independent_{metric}.png"); fig.savefig(output/f"independent_{metric}.pdf"); plt.close(fig)
    losses=root/"train/losses.csv"
    if losses.exists():
        with losses.open(encoding="utf-8") as stream: updates=list(csv.DictReader(stream))
        config=json.loads((root/"config.json").read_text(encoding="utf-8"))
        modern="target_q_mean" in updates[0]
        fig,axes=plt.subplots(4 if modern else 2,1,figsize=(8,10 if modern else 6),sharex=True)
        x=np.array([int(r["update"]) for r in updates])
        def curve(ax,key,label,**options):
            y=np.array([float(r[key]) for r in updates])
            ax.plot(x,y,alpha=.20,lw=.55,**options)
            ax.plot(x,[y[max(0,i-49):i+1].mean() for i in range(len(y))],lw=1.3,label=label,**options)
        curve(axes[0],"td_loss","TD loss"); axes[0].set_ylabel("Smooth L1 loss")
        curve(axes[1],"q_mean","Online Q")
        if modern:
            curve(axes[1],"target_q_mean","Target Q",linestyle="--")
            curve(axes[2],"td_target_mean","TD target")
            curve(axes[2],"td_error_mean","TD error",linestyle="--")
            curve(axes[3],"grad_norm","Gradient norm")
            axes[2].set_ylabel("Value"); axes[3].set_ylabel("L2 norm")
        axes[1].set_ylabel("Mean Q")
        tc=config["training"]
        if tc.get("target_update_mode","hard")=="hard":
            syncs=range(tc["target_update_steps"],int(x[-1])+1,tc["target_update_steps"])
            for ax in axes:
                for sync in syncs: ax.axvline(sync,color="#777777",ls=":",lw=.6,alpha=.55)
        for ax in axes:
            ax.grid(alpha=.2); ax.legend(fontsize=7,loc="best")
        axes[-1].set_xlabel("Gradient update")
        fig.suptitle("Raw traces (faint), trailing 50-update mean (solid); vertical lines: hard target sync")
        fig.tight_layout(); fig.savefig(output/"training_td_q.png"); fig.savefig(output/"training_td_q.pdf"); plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--run",type=Path,required=True)
    plot_monitoring(parser.parse_args().run.resolve())
