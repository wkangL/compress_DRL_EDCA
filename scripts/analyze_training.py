"""Audit every retained seed, summarize action usage and representation variance."""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import argparse,csv,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--run",type=Path,required=True); args=parser.parse_args()
    rows=[]; diagnostics=[]
    config=json.loads((args.run/"config.json").read_text())
    for folder in sorted(args.run.glob("*_seed*/train")):
        if not (folder/"steps.csv").exists(): continue
        with (folder/"steps.csv").open() as f: steps=list(csv.DictReader(f))
        actions=[json.loads(row["action"]) for row in steps]
        for i,branch in enumerate(("profile","alpha","beta1","beta2","gamma")):
            values=list(range(16)) if i==0 else config["action"][branch]
            count=Counter(a[i] for a in actions)
            for index,value in enumerate(values):
                rows.append(dict(run=folder.parent.name,branch=branch,value=value,count=count[index],fraction=count[index]/max(len(actions),1)))
        losses=[]
        if (folder/"losses.csv").exists():
            with (folder/"losses.csv").open() as f: losses=list(csv.DictReader(f))
        finite=all(np.isfinite(float(row[k])) for row in losses for k in ("td_loss","q_mean","q_min","q_max","grad_norm"))
        variances=np.array([json.loads(row["latent_variance"]) for row in losses])
        recent=variances[-min(100,len(variances)):].mean(axis=0) if len(variances) else np.array([])
        collapsed=np.flatnonzero(recent<config["model"]["collapse_variance"]).tolist()
        compressed=folder.parent.name.startswith("compressed")
        diagnostics.append(dict(run=folder.parent.name,transitions=len(steps),updates=len(losses),finite=finite,
                                last100_mean_variance=recent.tolist(),low_variance_dimensions=collapsed,
                                representation="learned_latent" if compressed else "raw_history",
                                latent_collapse_warning=bool(collapsed) if compressed else None,
                                note="Raw-history constant features are not encoder collapse. Low variance never authorizes discarding seeds."))
    out=ROOT/"results/summary"/args.run.name; out.mkdir(parents=True,exist_ok=True)
    if rows:
        with (out/"action_distribution.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    (out/"training_diagnostics.json").write_text(json.dumps(diagnostics,indent=2),encoding="utf-8")
    print(json.dumps(diagnostics,indent=2))


if __name__=="__main__": main()
