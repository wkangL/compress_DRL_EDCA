"""Plot recorded TD losses per seed; loss decrease alone is not convergence."""
from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT
from scripts.make_figures import METHODS,LABELS


def smooth(values,window=20):
    return np.array([values[max(0,i-window+1):i+1].mean() for i in range(len(values))])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args(); root=args.run.resolve()
    config=json.loads((root/"config.json").read_text(encoding="utf-8"))
    output=ROOT/"results/figures"/root.name; output.mkdir(parents=True,exist_ok=True)
    summary_dir=ROOT/"results/summary"/root.name; summary_dir.mkdir(parents=True,exist_ok=True)
    data={}; records=[]
    for folder in sorted(root.glob("*_seed*/train")):
        with (folder/"losses.csv").open(encoding="utf-8") as f: losses=list(csv.DictReader(f))
        with (folder/"steps.csv").open(encoding="utf-8") as f: steps=list(csv.DictReader(f))
        with (folder/"episodes.csv").open(encoding="utf-8") as f: episodes=list(csv.DictReader(f))
        if not losses: continue
        metadata=json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
        values=np.array([float(r["td_loss"]) for r in losses])
        updates=np.array([int(r["update"]) for r in losses])
        if not np.isfinite(values).all(): raise ValueError(f"Nonfinite TD loss in {folder}")
        method=metadata["method"]; seed=metadata["seed"]
        data.setdefault(method,[]).append((seed,updates,values))
        width=min(50,len(values))
        records.append(dict(method=method,seed=seed,episodes=len(episodes),transitions=len(steps),
            updates=len(values),target_syncs=int(metadata.get("target_updates",
                int(updates[-1])//config["training"]["target_update_steps"])),
            epsilon_start=float(steps[0]["epsilon"]),epsilon_last=float(steps[-1]["epsilon"]),
            loss_first50=float(values[:width].mean()),loss_last50=float(values[-width:].mean()),
            loss_max=float(values.max()),train_runtime_s=metadata["runtime_s"]))
    if not records: raise ValueError("No measured loss rows")
    with (summary_dir/"learning_budget_audit.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=records[0]); writer.writeheader(); writer.writerows(records)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,"savefig.dpi":300})

    def draw(ax,method):
        for seed,updates,values in data.get(method,[]):
            color=f"C{len(ax.lines)//2}"
            ax.plot(updates,values,color=color,alpha=.20,lw=.65)
            ax.plot(updates,smooth(values),color=color,lw=1.6,label=f"Train seed {seed} (MA20)")
        ax.set_xlabel("Gradient update"); ax.set_ylabel("Q / TD loss (Smooth L1)")
        ax.grid(alpha=.2); ax.spines[["top","right"]].set_visible(False); ax.legend(fontsize=8)
        ax.set_title(LABELS[METHODS.index(method)])

    primary="compressed_async_age"
    if primary in data:
        fig,ax=plt.subplots(figsize=(8,4)); draw(ax,primary)
        syncs=[r["target_syncs"] for r in records if r["method"]==primary]
        fig.text(.5,.015,f"Faint: raw minibatch loss; solid: trailing 20-update mean. Target syncs per seed: {syncs}.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.06,1,1))
        for extension in ("png","pdf"): fig.savefig(output/f"q_loss_main.{extension}")
        plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(11,10))
    for ax,method in zip(axes.flat,METHODS[1:]): draw(ax,method)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.05,.8,"Loss decrease does not prove policy convergence.\n\n"
        "Check independent evaluation over training,\nmultiple target-network synchronizations,\nseed variability and adequate exploration.\n\n"
        "Faint curves retain every recorded minibatch.\nMA20 is for visualization only.",va="top",fontsize=11)
    fig.tight_layout()
    for extension in ("png","pdf"): fig.savefig(output/f"q_loss_all.{extension}")
    plt.close(fig)
    (output/"q_loss_provenance.json").write_text(json.dumps(dict(run=str(root),source="*/train/losses.csv",
        loss="Smooth L1 across the five action branches",smoothing="trailing 20 updates",raw_data_modified=False,
        convergence_confirmed=False),indent=2),encoding="utf-8")
    print(output/"q_loss_main.png")
    print(json.dumps(records,indent=2))


if __name__=="__main__": main()
