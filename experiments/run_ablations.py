"""One requested ablation at a time; zero-shot and retrained outputs use separate paths."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT,load_config,merge
from rl.trainer import train
from rl.evaluator import evaluate
from rl.gates import require_edca_validation


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/quick.yaml")
    p.add_argument("--kind",required=True,choices=["decision_period","history_window","latent_dimension","coexistence","robustness"])
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--seed",type=int,default=1)
    p.add_argument("--method",default="compressed_async_age")
    args=p.parse_args(); require_edca_validation(); c=load_config(ROOT/args.config)
    train_again=args.kind in ("history_window","latent_dimension")
    if not train_again and not args.checkpoint: p.error("Zero-shot ablation requires --checkpoint")
    definitions={"decision_period":[(str(v),{"timing":{"decision_s":v}}) for v in c["ablations"]["decision_s"]],
                 "history_window":[(str(v),{"timing":{"history_s":v}}) for v in c["ablations"]["history_s"]],
                 "latent_dimension":[(str(v),{"model":{"latent_dim":v}}) for v in c["ablations"]["latent_dim"]],
                 "coexistence":[(str(v),{"network":{"enhanced_fraction":v}}) for v in c["ablations"]["enhanced_fraction"]],
                 "robustness":[("visible",{"network":{"hidden_nodes":False,"rts_cts":False}}),
                               ("hidden",{"network":{"hidden_nodes":True,"rts_cts":False}}),
                               ("hidden_rts",{"network":{"hidden_nodes":True,"rts_cts":True}})]}
    root=ROOT/"results/raw"/datetime.now().strftime(args.kind+"_%Y%m%d_%H%M%S"); root.mkdir(parents=True)
    mode="retrained" if train_again else "zero_shot"
    manifest=dict(kind=args.kind,mode=mode,checkpoint=str(args.checkpoint),seed=args.seed,status="running",completed=[])
    for label,override in definitions[args.kind]:
        cfg=merge(c,override); path=root/f"{mode}_{label}"
        checkpoint=train(cfg,args.method,args.seed,path/"train") if train_again else args.checkpoint
        evaluate(cfg,args.method,args.seed,checkpoint,path/"eval")
        manifest["completed"].append(label)
        (root/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    manifest["status"]="complete"; (root/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(root)


if __name__=="__main__": main()
