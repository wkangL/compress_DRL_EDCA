"""CCDFs and baseline comparisons from complete measured evaluation directories."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT,load_config
from reporting.comparison import ACS,relative_change,load_delays,balanced_ccdf
from rl.artifacts import save_json
from scripts.make_figures import METHODS,LABELS

PAIR=("default","compressed_async_age")
SCALARS={"goodput_mbps":("Goodput (Mbps)","higher"),"collision_rate":("Collision rate","lower"),
         "drop_rate":("Drop rate","lower"),"jain_fairness":("Jain fairness","higher"),
         "reward":("Reward","higher"),"report_age_mean_s":("Report age (s)","lower"),
         "fp32_equivalent_bytes_per_s":("FP32 equivalent bytes/s","lower")}
for ac in ACS:
    SCALARS[f"{ac}_goodput_mbps"]=(f"{ac.upper()} goodput (Mbps)","higher")
    SCALARS[f"{ac}_delay_mean_ms"]=(f"{ac.upper()} mean delay (ms)","lower")
    SCALARS[f"{ac}_delay_p95_ms"]=(f"{ac.upper()} p95 delay (ms)","lower")


def read_evaluations(paths):
    rows=[]; identities=set(); reference=None
    for path in paths:
        metadata=json.loads((path/"metadata.json").read_text(encoding="utf-8"))
        config=json.loads((path/"config.json").read_text(encoding="utf-8"))
        comparable={k:config[k] for k in ("network","queues","timing","traffic","evaluation")}
        comparable["source_hash"]=metadata["source_hash"]
        if reference is not None and comparable!=reference:
            raise ValueError("Comparison requires identical simulation and evaluation definitions")
        reference=comparable
        scenarios=load_config(ROOT/config["traffic"]["scenarios"])
        with (path/"evaluation.csv").open(encoding="utf-8") as f: batch=list(csv.DictReader(f))
        expected={(s,int(seed)) for s in scenarios for seed in config["evaluation"]["seeds"]}
        if len(batch)!=len(expected) or {(r["scenario"],int(r["eval_seed"])) for r in batch}!=expected:
            raise ValueError(f"Incomplete or duplicate evaluation rows: {path}")
        for row in batch:
            key=tuple(row[k] for k in ("method","train_seed","eval_seed","scenario"))
            if key in identities: raise ValueError("Duplicate evaluation replicate")
            if row["method"]!=metadata["method"] or int(row["train_seed"])!=metadata["seed"]:
                raise ValueError("Evaluation identity mismatch")
            identities.add(key); row["_directory"]=path; rows.append(row)
    return rows


def summarize_scalar(rows,metric):
    from scipy.stats import t
    grouped=defaultdict(list)
    for row in rows:
        if metric not in row: continue
        if "_delay_" in metric and row.get(metric.split('_')[0]+"_delay_valid") not in ("True","1","1.0"):
            continue
        value=float(row[metric])
        if not np.isfinite(value): raise ValueError(f"Nonfinite {metric}")
        seed=row["eval_seed"] if row["method"]=="default" else row["train_seed"]
        grouped[seed].append(value)
    values=np.asarray([np.mean(v) for v in grouped.values()])
    if not len(values): return None,None,0
    ci=float(t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
    return float(values.mean()),ci,len(values)


def build_comparisons(paths,output,summary):
    rows=read_evaluations(paths); output.mkdir(parents=True,exist_ok=True); summary.mkdir(parents=True,exist_ok=True)
    pair_dir=output/"baseline_vs_compressed"; pair_dir.mkdir(exist_ok=True)
    scenarios=sorted({r["scenario"] for r in rows}); methods=[m for m in METHODS if any(r["method"]==m for r in rows)]
    lookup={}; records=[]
    for scenario in scenarios:
        for metric,(label,direction) in SCALARS.items():
            for method in methods:
                subset=[r for r in rows if r["scenario"]==scenario and r["method"]==method]
                lookup[method,scenario,metric]=summarize_scalar(subset,metric)
            baseline=lookup.get(("default",scenario,metric),(None,None,0))[0]
            for method in methods:
                value,ci,n=lookup[method,scenario,metric]
                delta=relative_change(baseline,value) if baseline is not None and value is not None else None
                benefit=delta if direction=="higher" else -delta if delta is not None else None
                records.append(dict(scenario=scenario,method=method,metric=metric,baseline=baseline,value=value,
                    ci95=ci,n=n,change_pct=delta,improvement_pct=benefit,direction=direction,
                    note="undefined percentage" if delta is None else "ratio of means; not significance"))
    for name,selected in (("comparison_percentages.csv",records),
                          ("baseline_vs_compressed.csv",[r for r in records if r["method"]==PAIR[1]])):
        with (summary/name).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=records[0]); writer.writeheader(); writer.writerows(selected)
    lines=["# Default EDCA 与 Compressed Async + Age 对比","",
        "相对变化=(完整方案−baseline)/|baseline|×100%；改善率按指标方向调整，正值改善、负值恶化。baseline为0或时延无成功样本时百分比为N/A。这里只比较实测均值，不表示统计显著。", "",
        "| 场景 | 指标 | baseline | 完整方案 | 相对变化 | 改善率 |","|---|---|---:|---:|---:|---:|"]
    fmt=lambda v:"N/A" if v is None else f"{v:.4g}"
    pct=lambda v:"N/A" if v is None else f"{v:+.4g}%"
    for r in records:
        if r["method"]==PAIR[1]:
            lines.append(f"|{r['scenario'].split('_')[0]}|{r['metric']}|{fmt(r['baseline'])}|{fmt(r['value'])}|{pct(r['change_pct'])}|{pct(r['improvement_pct'])}|")
    (summary/"baseline_vs_compressed.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,"savefig.dpi":300})
    colors={m:c for m,c in zip(METHODS,["#444444","#0072B2","#009E73","#D55E00","#CC79A7","#E69F00"])}
    def save(fig,directory,name):
        fig.savefig(directory/f"{name}.png",bbox_inches="tight"); fig.savefig(directory/f"{name}.pdf",bbox_inches="tight"); plt.close(fig)
    if all(m in methods for m in PAIR):
        for metric,(label,_) in SCALARS.items():
            if "_delay_" in metric: continue  # Delay graphics are exclusively CCDFs.
            fig,ax=plt.subplots(figsize=(7,3.7)); x=np.arange(len(scenarios))
            for i,method in enumerate(PAIR):
                vals=[lookup[method,s,metric] for s in scenarios]
                ax.errorbar(x+(i-.5)*.04,[v[0] for v in vals],yerr=[v[1] or 0 for v in vals],
                    color=colors[method],marker="o" if i==0 else "s",capsize=3,label=LABELS[METHODS.index(method)])
            ax.set_xticks(x,[s.split('_')[0] for s in scenarios]); ax.set_ylabel(label); ax.set_xlabel("Traffic scenario")
            ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout(); save(fig,pair_dir,metric)

    distributions=defaultdict(lambda:defaultdict(dict)); counts=[]
    for row in rows:
        episode=row["_directory"]/f"{row['scenario']}_seed{row['eval_seed']}"
        for ac,samples in load_delays(episode).items():
            if not np.isfinite(samples).all() or (samples<0).any(): raise ValueError("Invalid delays")
            distributions[row["method"],row["scenario"],ac][row["train_seed"]][row["eval_seed"]]=samples
            counts.append({k:row[k] for k in ("method","scenario","train_seed","eval_seed")}|dict(ac=ac,successful_packets=len(samples)))
    with (summary/"delay_sample_counts.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=counts[0]); writer.writeheader(); writer.writerows(counts)
    curves=[]
    for ac in ACS:
        columns=2; rows_count=int(np.ceil(len(scenarios)/columns))
        fig_all,axes_all=plt.subplots(rows_count,columns,figsize=(10,3.5*rows_count),squeeze=False)
        fig_pair,axes_pair=plt.subplots(rows_count,columns,figsize=(10,3.5*rows_count),squeeze=False)
        for index,scenario in enumerate(scenarios):
            pooled=[v for m in methods for values in distributions[m,scenario,ac].values() for v in values.values() if len(v)]
            grid=np.unique(np.r_[0,np.quantile(np.concatenate(pooled),np.linspace(0,1,1200)),
                np.concatenate([np.sort(v)[-20:] for v in pooled])]) if pooled else np.array([0.])
            for method in methods:
                curve=balanced_ccdf(distributions[method,scenario,ac],grid)
                for x,y in zip(grid,curve):
                    curves.append(dict(method=method,scenario=scenario,ac=ac,delay_ms=x,probability_gt=y if np.isfinite(y) else None))
                # Log axis cannot display zero; retain true zeros in the numeric export.
                displayed=np.where(curve>0,curve,np.nan)
                for axes in ([axes_all,axes_pair] if method in PAIR else [axes_all]):
                    axes.flat[index].step(grid,displayed,where="post",color=colors[method],
                        linestyle="-" if method=="default" else "--",label=LABELS[METHODS.index(method)])
            for axes in (axes_all,axes_pair):
                ax=axes.flat[index]; ax.set_yscale("log"); ax.set_title(f"{scenario.split('_')[0]} / {ac.upper()}")
                ax.set_xlabel("MAC enqueue-to-ACK delay (ms)"); ax.set_ylabel("P(delay > x)"); ax.grid(alpha=.2,which="both")
        for fig,axes in ((fig_all,axes_all),(fig_pair,axes_pair)):
            for index in range(len(scenarios),axes.size): axes.flat[index].set_visible(False)
            handles,labels=axes.flat[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc="lower center",ncol=2,fontsize=8)
            fig.suptitle("Successful packets only; equal eval-seed then train-seed weighting",fontsize=10)
            fig.tight_layout(rect=(0,.13,1,.96))
        save(fig_all,output,f"delay_ccdf_{ac}"); save(fig_pair,pair_dir,f"delay_ccdf_{ac}")
    with (summary/"delay_ccdf.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=curves[0]); writer.writeheader(); writer.writerows(curves)
    save_json(output/"comparison_provenance.json",dict(evaluations=[str(p.resolve()) for p in paths],
        ccdf="P(D>x); successful packets only; equal evaluation seeds within train seed, then equal train seeds",
        grid="1200 pooled quantiles plus 20 largest delays from each replicate; probabilities exact at each x",
        empty_distribution="undefined; excluded and reported in delay_sample_counts.csv",zero_probability="retained in CSV, omitted on log axis",
        percentages="100*(candidate-baseline)/abs(baseline); undefined if baseline=0",raw_data_modified=False))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--run",type=Path,required=True); args=parser.parse_args()
    from scripts.audit_run import audit_run
    audit_run(args.run)
    build_comparisons(sorted(args.run.glob("*_seed*/eval")),ROOT/"results/figures"/args.run.name,ROOT/"results/summary"/args.run.name)
    print(ROOT/"results/figures"/args.run.name/"baseline_vs_compressed")


if __name__=="__main__": main()
