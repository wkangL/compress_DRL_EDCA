"""Compare DDQN target-network update diagnostics without claiming policy convergence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT


LABELS = {
    "hard500": "Hard / 500 updates",
    "soft005": "Soft / tau=0.005",
    "soft001": "Soft / tau=0.001",
}


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def f(row: dict, name: str) -> float:
    return float(row[name])


def smooth(values: np.ndarray, width: int = 20) -> np.ndarray:
    return np.asarray([values[max(0, i-width+1):i+1].mean() for i in range(len(values))])


def write_csv(path: Path, values: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard500", type=Path, required=True)
    parser.add_argument("--soft005", type=Path, required=True)
    parser.add_argument("--soft001", type=Path, required=True)
    parser.add_argument("--name", default="diagnostic_target_updates_20261020")
    args = parser.parse_args()
    runs = {key: getattr(args, key).resolve() for key in LABELS}
    summary_dir = ROOT/"results"/"summary"/args.name
    figure_dir = ROOT/"results"/"figures"/args.name
    summary_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)

    loaded = {}; summaries = []; evaluations = []; replay_rows = []
    for key, run in runs.items():
        progress = json.loads((run/"progress.json").read_text(encoding="utf-8"))
        if progress["state"] != "complete" or progress["updates"] != 500:
            raise ValueError(f"Incomplete diagnostic: {run}")
        losses = rows(run/"train"/"losses.csv")
        evaluation = rows(run/"monitoring"/"learning_curve.csv")
        replay = rows(run/"train"/"replay_audit.csv")[-1]
        if len(losses) != 500 or len(evaluation) != 24:
            raise ValueError(f"Unexpected diagnostic budget: {run}")
        loaded[key] = losses
        first = losses[:50]; last = losses[-50:]
        summaries.append({
            "scheme": key, "label": LABELS[key], "updates": len(losses),
            "target_updates": progress["target_syncs"],
            "loss_first50_mean": np.mean([f(r,"td_loss") for r in first]),
            "loss_last50_mean": np.mean([f(r,"td_loss") for r in last]),
            "loss_max": max(f(r,"td_loss") for r in losses),
            "td_error_abs_last50_mean": np.mean([f(r,"td_error_abs_mean") for r in last]),
            "q_abs_max": max(max(abs(f(r,"q_min")),abs(f(r,"q_max"))) for r in losses),
            "target_q_abs_max": max(max(abs(f(r,"target_q_min")),abs(f(r,"target_q_max"))) for r in losses),
            "grad_norm_max": max(f(r,"grad_norm") for r in losses),
            "target_distance_max_before": max(f(r,"target_distance_before") for r in losses),
            "target_distance_final_after": f(losses[-1],"target_distance_after"),
            "eval_reward_stage1_mean": np.mean([f(r,"reward") for r in evaluation if int(r["stage"])==1]),
            "eval_reward_stage6_mean": np.mean([f(r,"reward") for r in evaluation if int(r["stage"])==6]),
        })
        for stage in range(1,7):
            stage_rows = [r for r in evaluation if int(r["stage"]) == stage]
            evaluations.append({"scheme":key,"stage":stage,
                "reward_mean":np.mean([f(r,"reward") for r in stage_rows]),
                "goodput_mbps_mean":np.mean([f(r,"goodput_mbps") for r in stage_rows]),
                "delivery_fraction_mean":np.mean([f(r,"delivery_fraction") for r in stage_rows])})
        scenario_counts = json.loads(replay["scenario_counts"])
        action_counts = json.loads(replay["action_branch_counts"])
        replay_rows.append({"scheme":key,"transitions":int(replay["transitions"]),
            "exploratory_fraction":float(replay["exploratory_fraction"]),
            "terminal_count":int(replay["terminal_count"]),"truncated_count":int(replay["truncated_count"]),
            "scenario_counts":json.dumps(scenario_counts,ensure_ascii=False,sort_keys=True),
            "action_unique_counts":json.dumps([len(branch) for branch in action_counts])})

    write_csv(summary_dir/"stability_summary.csv", summaries)
    write_csv(summary_dir/"independent_evaluation_by_stage.csv", evaluations)
    write_csv(summary_dir/"replay_coverage.csv", replay_rows)
    provenance = {"purpose":"short target-update diagnostic; not a convergence or formal performance result",
        "fixed_budget":{"train_seed":11,"eval_seed":301,"updates":500,"transitions":120,"stages":6},
        "runs":{key:str(path) for key,path in runs.items()},
        "loss":"Smooth L1 TD loss over five action branches", "smoothing":"trailing 20 updates; raw CSV retained"}
    (summary_dir/"provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,"savefig.dpi":300})
    colors={"hard500":"#1f77b4","soft005":"#d62728","soft001":"#2ca02c"}
    fig, axes = plt.subplots(2,2,figsize=(11,7.5))
    for key, losses in loaded.items():
        x=np.asarray([int(r["update"]) for r in losses]); c=colors[key]; label=LABELS[key]
        loss=np.asarray([f(r,"td_loss") for r in losses])
        q=np.asarray([f(r,"q_mean") for r in losses]); tq=np.asarray([f(r,"target_q_mean") for r in losses])
        err=np.asarray([f(r,"td_error_abs_mean") for r in losses])
        dist=np.asarray([f(r,"target_distance_before") for r in losses])
        axes[0,0].plot(x,smooth(loss),color=c,label=label)
        axes[0,1].plot(x,smooth(q),color=c,label=f"{label}: online")
        axes[0,1].plot(x,smooth(tq),color=c,ls="--",alpha=.8,label=f"{label}: target")
        axes[1,0].plot(x,smooth(err),color=c,label=label)
        axes[1,1].plot(x,dist,color=c,label=label)
    axes[0,0].set(title="TD loss (MA20)",ylabel="Smooth L1 loss")
    axes[0,1].set(title="Selected online and target Q (MA20)",ylabel="Mean Q")
    axes[1,0].set(title="Absolute TD error (MA20)",ylabel="Mean |TD error|",xlabel="Gradient update")
    axes[1,1].set(title="Online-target parameter RMS distance",ylabel="RMS distance",xlabel="Gradient update")
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.spines[["top","right"]].set_visible(False); ax.legend(fontsize=7)
    fig.suptitle("DDQN target-update diagnostic (same seed and sampling budget)")
    fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(figure_dir/f"target_update_stability.{ext}")
    plt.close(fig)

    fig, axes = plt.subplots(1,3,figsize=(12,3.5))
    for key in LABELS:
        current=[r for r in evaluations if r["scheme"]==key]
        x=[r["stage"] for r in current]; c=colors[key]
        axes[0].plot(x,[r["reward_mean"] for r in current],marker="o",color=c,label=LABELS[key])
        axes[1].plot(x,[r["goodput_mbps_mean"] for r in current],marker="o",color=c,label=LABELS[key])
        axes[2].plot(x,[r["delivery_fraction_mean"] for r in current],marker="o",color=c,label=LABELS[key])
    for ax,title,ylabel in zip(axes,("Independent reward","Independent goodput","Independent delivery fraction"),("Mean reward","Mbit/s","Fraction")):
        ax.set(xlabel="Training stage",ylabel=ylabel,title=title); ax.set_xticks(range(1,7)); ax.grid(alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(figure_dir/f"independent_evaluation.{ext}")
    plt.close(fig)
    print(json.dumps(summaries,ensure_ascii=False,indent=2))
    print(figure_dir)


if __name__ == "__main__":
    main()
