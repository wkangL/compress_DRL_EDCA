"""IEEE-style plots from immutable evaluation CSVs, with seed-level Student-t intervals."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT

METHODS=["default","full_fresh","raw_async","compressed_fresh","compressed_async_no_age","compressed_async_age"]
LABELS=["Default EDCA","Full-State Fresh","Raw Async","Compressed Fresh","Compressed Async No-Age","Compressed Async + Age"]
METRICS={"goodput_mbps":("goodput_main","Goodput (Mbps)"),"collision_rate":("collision_main","Collision rate"),
         "drop_rate":("drop_main","Drop rate"),"jain_fairness":("fairness_main","Jain fairness"),
         "vo_delay_p95_ms":("vo_delay_main","VO p95 delay (ms)"),"vi_delay_p95_ms":("vi_delay_main","VI p95 delay (ms)"),
         "report_age_mean_s":("report_age","Mean report age (s)"),"fp32_equivalent_bytes_per_s":("feedback_overhead","FP32 equivalent bytes/s")}


def collect(root: Path) -> list[dict]:
    from scripts.audit_run import audit_run
    audit_run(root)
    status_path=root/"status.json"
    if status_path.exists():
        status=json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state")!="complete":
            raise ValueError("Run is incomplete or has failed seeds; do not silently summarize only successful jobs")
    rows=[]
    for path in sorted(root.glob("*_seed*/eval/evaluation.csv")):
        with path.open(encoding="utf-8") as f: rows.extend(csv.DictReader(f))
    if not rows: raise ValueError(f"No evaluation results in {root}")
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    from scipy.stats import t
    grouped=defaultdict(lambda:defaultdict(list))
    for row in rows:
        # Baseline has no training randomness. Avoid multiplying identical default runs.
        replicate=row["eval_seed"] if row["method"]=="default" else row["train_seed"]
        for metric in METRICS:
            if metric not in row: continue
            grouped[(row["method"],row["scenario"],metric)][replicate].append(float(row[metric]))
    output=[]
    for (method,scenario,metric),seeds in sorted(grouped.items()):
        values=np.array([np.mean(v) for v in seeds.values()]); n=len(values)
        ci=float(t.ppf(.975,n-1)*values.std(ddof=1)/np.sqrt(n)) if n>1 else None
        output.append(dict(method=method,scenario=scenario,metric=metric,mean=float(values.mean()),ci95=ci,n=n))
    return output


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--run",type=Path,required=True); args=parser.parse_args()
    root=args.run.resolve(); rows=collect(root); summary=summarize(rows)
    out=ROOT/"results/figures"/root.name; out.mkdir(parents=True,exist_ok=True)
    summary_dir=ROOT/"results/summary"/root.name; summary_dir.mkdir(parents=True,exist_ok=True)
    with (summary_dir/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager
    available={f.name for f in fontManager.ttflist}
    font=next((f for f in ("Times New Roman","Liberation Serif","DejaVu Serif") if f in available),"DejaVu Serif")
    plt.rcParams.update({"font.family":font,"font.size":10,"axes.labelsize":11,"xtick.labelsize":9.5,
                         "ytick.labelsize":9.5,"legend.fontsize":9,"lines.linewidth":1.6,"savefig.dpi":300,
                         "pdf.fonttype":42,"ps.fonttype":42,"figure.facecolor":"white","axes.facecolor":"white"})
    scenarios=sorted({r["scenario"] for r in rows}); lookup={(r["method"],r["scenario"],r["metric"]):r for r in summary}
    colors=["#444444","#0072B2","#009E73","#D55E00","#CC79A7","#E69F00"]
    markers=["o","s","^","D","v","P"]; styles=["-","--","-.",":","--","-"]
    for metric,(filename,label) in METRICS.items():
        if "_delay_" in metric: continue  # Packet delay plots are produced as CCDFs.
        fig,ax=plt.subplots(figsize=(7,3.8)); x=np.arange(len(scenarios))
        for i,(method,title) in enumerate(zip(METHODS,LABELS)):
            if not all((method,s,metric) in lookup for s in scenarios): continue
            records=[lookup[method,s,metric] for s in scenarios]
            ax.errorbar(x+(i-2.5)*.025,[r["mean"] for r in records],
                        yerr=[r["ci95"] or 0 for r in records],color=colors[i],marker=markers[i],
                        linestyle=styles[i],label=title,capsize=2,markersize=4.5)
        ax.set_xticks(x,[s.split("_")[0] for s in scenarios]); ax.set_xlabel("Traffic scenario")
        ax.set_ylabel(label); ax.grid(axis="y",alpha=.22); ax.spines[["top","right"]].set_visible(False)
        ax.legend(loc="upper center",bbox_to_anchor=(.5,-.22),ncol=3,frameon=False)
        fig.tight_layout(); fig.savefig(out/f"{filename}.pdf",bbox_inches="tight"); fig.savefig(out/f"{filename}.png",bbox_inches="tight"); plt.close(fig)
    # Preserve individual training seeds; smoothing only adds a visual trend.
    fig,ax=plt.subplots(figsize=(7,3.5))
    for i,method in enumerate(METHODS[1:],1):
        series=[]
        for path in sorted(root.glob(f"{method}_seed*/train/episodes.csv")):
            with path.open() as f: vals=np.array([float(r["reward"]) for r in csv.DictReader(f)])
            if len(vals): series.append(vals); ax.plot(np.arange(len(vals))+1,vals,color=colors[i],alpha=.18,lw=.7)
        if series:
            mean=np.mean(series,axis=0); trend=np.array([mean[max(0,j-3):j+1].mean() for j in range(len(mean))])
            ax.plot(np.arange(len(trend))+1,trend,color=colors[i],linestyle=styles[i],label=LABELS[i])
    ax.set_xlabel("Training episode"); ax.set_ylabel("Mean episode reward"); ax.grid(alpha=.2)
    ax.legend(fontsize=8,ncol=2); fig.tight_layout()
    fig.savefig(out/"training_reward.pdf"); fig.savefig(out/"training_reward.png"); plt.close(fig)
    (out/"provenance.json").write_text(json.dumps(dict(run=str(root),font=font,ci="Student t, 95%; AI: train seed after averaging evaluation seeds; Default: evaluation seed",raw_data_modified=False),indent=2),encoding="utf-8")
    print(out)


if __name__=="__main__": main()
