"""Aggregate the three-seed Reward-v2 medium-budget monitoring runs."""
from pathlib import Path
import csv,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT

SEEDS=(11,22,33); STAGES=(1,4,8,12,16,20)
ROOTS={s:ROOT/f"results/raw/reward_v2_medium_seed{s}_20261021" for s in SEEDS}
OUT=ROOT/"results/summary/reward_v2_medium_3seed_20261021"
FIG=ROOT/"results/figures/reward_v2_medium_3seed_20261021"

def read(path):
    with path.open(encoding="utf-8") as f: return list(csv.DictReader(f))

def service(row,ac):
    offered=float(row[f"{ac}_offered_mbps"]); return min(float(row[f"{ac}_goodput_mbps"])/max(offered,1e-12),1.) if offered else 1.

def main():
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True); rows=[]
    for seed,root in ROOTS.items():
        for stage in STAGES:
            for r in read(root/f"monitoring/stage_{stage:03d}/eval/evaluation.csv"):
                rows.append(dict(train_seed=seed,stage=stage,scenario=r["scenario"],eval_seed=int(r["eval_seed"]),
                    reward=float(r["reward"]),goodput_mbps=float(r["goodput_mbps"]),collision_rate=float(r["collision_rate"]),
                    drop_rate=float(r["drop_rate"]),vo_on_time=float(r["vo_on_time_delivery_rate"]),
                    vi_on_time=float(r["vi_on_time_delivery_rate"]),vo_p95_ms=float(r["vo_delay_p95_ms"]),
                    vi_p95_ms=float(r["vi_delay_p95_ms"]),be_service=service(r,"be"),bk_service=service(r,"bk")))
    with (OUT/"independent_evaluation_by_stage.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    diagnostics=[]
    for seed,root in ROOTS.items():
        loss=read(root/"train/losses.csv"); arrays={k:np.array([float(r[k]) for r in loss]) for k in
            ("td_loss","q_mean","target_q_mean","td_error_abs_mean","grad_norm")}
        latent=np.array([json.loads(r["latent_variance"]) for r in loss])
        diagnostics.append(dict(seed=seed,updates=len(loss),finite=all(np.isfinite(v).all() for v in arrays.values()),
            td_loss_first200=float(arrays["td_loss"][:200].mean()),td_loss_last200=float(arrays["td_loss"][-200:].mean()),
            q_last200=float(arrays["q_mean"][-200:].mean()),target_q_last200=float(arrays["target_q_mean"][-200:].mean()),
            td_error_abs_last200=float(arrays["td_error_abs_mean"][-200:].mean()),grad_norm_max=float(arrays["grad_norm"].max()),
            latent_variance_last200_min=float(latent[-200:].mean(0).min()),collapsed_dimensions=int((latent[-200:].mean(0)<1e-5).sum())))
    (OUT/"training_diagnostics.json").write_text(json.dumps(diagnostics,indent=2),encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors={11:"#0072B2",22:"#D55E00",33:"#009E73"}; scenarios=sorted({r["scenario"] for r in rows})
    fig,axes=plt.subplots(3,1,figsize=(9,9),sharex=True)
    for seed,root in ROOTS.items():
        loss=read(root/"train/losses.csv"); x=np.array([int(r["update"]) for r in loss]); y=np.array([float(r["td_loss"]) for r in loss]); w=50
        axes[0].plot(x[w-1:],np.convolve(y,np.ones(w)/w,"valid"),color=colors[seed],label=f"seed {seed}")
        axes[1].plot(x,[float(r["q_mean"]) for r in loss],color=colors[seed],alpha=.55)
        axes[2].plot(x,[float(r["target_q_mean"]) for r in loss],color=colors[seed],alpha=.55)
    axes[0].set_ylabel("TD loss (50-update mean)"); axes[1].set_ylabel("Mean Q"); axes[2].set_ylabel("Mean target Q"); axes[2].set_xlabel("Gradient update")
    for ax in axes: ax.grid(alpha=.2)
    axes[0].legend(); fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(FIG/f"medium_training_diagnostics.{ext}",bbox_inches="tight")
    plt.close(fig)

    fig,axes=plt.subplots(len(scenarios),3,figsize=(11,2.5*len(scenarios)),sharex=True)
    for i,scenario in enumerate(scenarios):
        for seed in SEEDS:
            subset=[r for r in rows if r["scenario"]==scenario and r["train_seed"]==seed]
            x=STAGES
            axes[i,0].plot(x,[np.mean([r["vi_on_time"] for r in subset if r["stage"]==s]) for s in x],marker="o",color=colors[seed])
            axes[i,1].plot(x,[np.mean([r["vi_p95_ms"] for r in subset if r["stage"]==s]) for s in x],marker="o",color=colors[seed])
            axes[i,2].plot(x,[np.mean([r["be_service"] for r in subset if r["stage"]==s]) for s in x],marker="o",color=colors[seed],label=f"seed {seed}")
        axes[i,0].set_ylabel(f"{scenario.split('_')[0]}\nVI on-time"); axes[i,1].set_ylabel("VI p95 (ms)"); axes[i,2].set_ylabel("BE service")
        for ax in axes[i]: ax.grid(alpha=.2)
    for ax in axes[-1]: ax.set_xlabel("Training stage")
    axes[0,2].legend(); fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(FIG/f"medium_independent_qos.{ext}",bbox_inches="tight")
    plt.close(fig)
    print(OUT)

if __name__=="__main__": main()
