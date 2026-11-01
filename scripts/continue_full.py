"""Train or resume one full seed, monitor stages, then run held-out final tests."""
from pathlib import Path
from datetime import datetime
import argparse,json,os,sys,traceback,zipfile,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT,load_config
from rl.artifacts import save_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/full.yaml")
    p.add_argument("--monitor-config",default="configs/monitoring.json")
    p.add_argument("--resume",type=Path)
    p.add_argument("--method",default="compressed_async_age",
                   choices=["full_fresh","raw_async","compressed_fresh","compressed_async_no_age","compressed_async_age"])
    p.add_argument("--seed",type=int,default=1)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(); config=load_config(ROOT/args.config)
    options=json.loads((ROOT/args.monitor_config).read_text(encoding="utf-8"))
    import torch
    from rl.trainer import train
    from rl.evaluator import evaluate
    from rl.monitoring import StageMonitor
    from scripts.make_comparisons import build_comparisons
    if args.resume:
        saved=torch.load(args.resume,map_location="cpu",weights_only=False)
        seed=saved["seed"]; method=saved["method"]; start=saved["stage"]+1
    else:
        saved=dict(stage=-1,steps=0,updates=0); seed=args.seed; method=args.method; start=1
    root=args.output.resolve(); root.mkdir(parents=True,exist_ok=False)
    save_json(root/"config.json",config)
    save_json(root/"launch.json",dict(pid=os.getpid(),checkpoint=str(args.resume.resolve()) if args.resume else None,method=method,seed=seed,
        started_at=datetime.now().astimezone().isoformat(),monitoring=options,command=sys.argv))
    files=[ROOT/"run.py",ROOT/"project_config.py"]
    for directory in ("rl","reporting","models","sim","scripts","configs"):
        files.extend(f for f in (ROOT/directory).rglob('*') if f.is_file() and '__pycache__' not in f.parts)
    with zipfile.ZipFile(root/"source_snapshot.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for file in files: archive.write(file,file.relative_to(ROOT))
    save_json(root/"source_sha256.json",{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    monitor=StageMonitor(root,config,options,start)
    monitor.progress("starting",saved)
    del saved
    try:
        checkpoint=train(config,method,seed,root/"train",args.resume,on_stage=monitor)
        # Release large replay objects before the final test; evaluator loads only a compact snapshot.
        snapshot=root/"snapshots"/f"stage_{config['training']['stages']:03d}.pt"
        final_state=torch.load(snapshot,map_location="cpu",weights_only=False)
        if options.get("final_test",True):
            monitor.progress("final_testing",final_state)
            final=root/"final"; final.mkdir(); save_json(final/"config.json",config)
            jobs=[]
            planned=[dict(method=m,seed=seed) for m in ("default",method)]
            save_json(final/"status.json",dict(state="running",planned_jobs=planned,jobs=jobs))
            paths=[]
            for name in ("default",method):
                directory=final/f"{name}_seed{seed}"/"eval"; paths.append(directory)
                rows=evaluate(config,name,seed,None if name=="default" else snapshot,directory)
                jobs.append(dict(status="complete",method=name,seed=seed,rows=len(rows)))
                save_json(final/"status.json",dict(state="running",planned_jobs=planned,jobs=jobs))
            save_json(final/"status.json",dict(state="complete",planned_jobs=planned,jobs=jobs))
            build_comparisons(paths,ROOT/"results/figures"/root.name/"final",ROOT/"results/summary"/root.name/"final")
        monitor.progress("complete",final_state)
        with (ROOT/"reports/实验进展.md").open("a",encoding="utf-8") as stream:
            stream.write(f"\n- 自动进展 {datetime.now().astimezone().isoformat()}：`{root.name}`完整方案seed {seed}训练完成，"
                "最终独立测试及baseline对比按monitoring计划执行完成。仅单训练seed，仍不宣称收敛或显著优越。\n")
        print(root,flush=True)
    except BaseException as error:
        state=json.loads((root/"progress.json").read_text(encoding="utf-8"))
        state.update(state="failed",error=repr(error),failed_at=datetime.now().astimezone().isoformat())
        save_json(root/"progress.json",state)
        with (ROOT/"reports/实验进展.md").open("a",encoding="utf-8") as stream:
            stream.write(f"\n- 自动进展：`{root.name}`运行中止：{error!r}。保留原始数据和最新完整checkpoint，未标记完成。\n")
        traceback.print_exc(); raise


if __name__=="__main__": main()
