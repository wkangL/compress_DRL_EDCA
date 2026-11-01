"""Stage-gated experiment driver. Every invocation writes a new immutable run directory."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime
from pathlib import Path
import argparse,json,subprocess,sys,time,shutil
from project_config import ROOT,load_config
from rl.artifacts import save_json


def worker(config: dict,method: str,seed: int,root: str,reuse: str|None=None) -> dict:
    from rl.trainer import train
    from rl.evaluator import evaluate
    folder=Path(root)/f"{method}_seed{seed}"
    checkpoint=None
    old=Path(reuse)/f"{method}_seed{seed}"/"train" if reuse else None
    if method!="default":
        if old and (old/"checkpoint.pt").exists() and (old/"metadata.json").exists():
            metadata=json.loads((old/"metadata.json").read_text(encoding="utf-8"))
            if metadata.get("state","complete")!="complete":
                raise ValueError("Paused training requires --resume, not --reuse-run")
            previous=json.loads((old/"config.json").read_text())
            # Worker count affects scheduling only, not research or random streams.
            previous["run"]["workers"]=config["run"]["workers"]
            if previous!=config: raise ValueError("Cannot reuse checkpoint with a different experiment configuration")
            (folder/"train").mkdir(parents=True)
            for name in ("checkpoint.pt","steps.csv","episodes.csv","losses.csv","config.json","metadata.json"):
                if (old/name).exists(): shutil.copy2(old/name,folder/"train"/name)
            (folder/"train/reused_from.json").write_text(json.dumps(dict(source=str(old.resolve())),indent=2))
            checkpoint=folder/"train/checkpoint.pt"
        else: checkpoint=train(config,method,seed,folder/"train")
    result=evaluate(config,method,seed,checkpoint,folder/"eval")
    return dict(method=method,seed=seed,checkpoint=str(checkpoint),rows=len(result))


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/quick.yaml")
    parser.add_argument("--phase",choices=["all","validate","train","evaluate"],default="all")
    parser.add_argument("--method",choices=["default","full_fresh","raw_async","compressed_fresh","compressed_async_no_age","compressed_async_age"])
    parser.add_argument("--seed",type=int)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--reuse-run",type=Path,help="Reuse completed checkpoints with identical research config")
    parser.add_argument("--resume",type=Path,help="Exact continuation from a stage-boundary checkpoint with replay/RNG")
    parser.add_argument("--stop-after-stage",type=int,help="Stop train at this one-based stage boundary; preserve the full research budget")
    args=parser.parse_args(); config=load_config(ROOT/args.config)
    if (args.resume or args.stop_after_stage is not None) and args.phase!="train":
        parser.error("--resume and --stop-after-stage require --phase train")
    if args.phase=="validate":
        subprocess.run([sys.executable,str(ROOT/"experiments/run_edca_validation.py")],check=True); return
    from rl.gates import require_edca_validation
    require_edca_validation()
    if not (ROOT/config["traffic"]["capacity_file"]).exists():
        subprocess.run([sys.executable,str(ROOT/"scripts/calibrate_capacity.py")],check=True)
    root=ROOT/config["run"]["output"]/datetime.now().strftime(config["run"]["mode"]+"_%Y%m%d_%H%M%S")
    root.mkdir(parents=True,exist_ok=False)
    (root/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    started=time.monotonic()
    if args.phase in ("train","evaluate"):
        if not args.method: parser.error("--method is required for train/evaluate")
        seed=args.seed if args.seed is not None else config["run"]["seeds"][0]
        if args.phase=="train":
            from rl.trainer import train
            if args.method=="default": parser.error("Default EDCA has no trainable model")
            print(train(config,args.method,seed,root/"train",args.resume,args.stop_after_stage))
        else:
            from rl.evaluator import evaluate
            evaluate(config,args.method,seed,args.checkpoint,root/"eval")
    else:
        methods=[args.method] if args.method else config["evaluation"]["methods"]
        seeds=[args.seed] if args.seed is not None else config["run"]["seeds"]
        jobs=[(m,s) for m in methods for s in (seeds[:1] if m=="default" else seeds)]
        status={"state":"running","planned_jobs":[dict(method=m,seed=s) for m,s in jobs],"jobs":[]}
        save_json(root/"status.json",status)
        with ProcessPoolExecutor(max_workers=config["run"]["workers"]) as pool:
            futures={pool.submit(worker,config,m,s,str(root),str(args.reuse_run) if args.reuse_run else None):(m,s) for m,s in jobs}
            for future in as_completed(futures):
                method,seed=futures[future]
                try: status["jobs"].append(dict(status="complete",**future.result()))
                except Exception as error:
                    status["jobs"].append(dict(status="failed",method=method,seed=seed,error=repr(error)))
                save_json(root/"status.json",status)
        status["state"]="complete" if all(j["status"]=="complete" for j in status["jobs"]) else "failed"
        status["runtime_s"]=time.monotonic()-started
        save_json(root/"status.json",status)
        if status["state"]!="complete": raise SystemExit(f"Some jobs failed; see {root/'status.json'}")
    print(root)


if __name__=="__main__": main()
