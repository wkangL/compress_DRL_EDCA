"""Stage monitoring with independent rollouts and no training RNG side effects."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import csv,json,math,random
import numpy as np
import torch
from project_config import ROOT
from rl.artifacts import atomic_output,save_json
from rl.training_budget import training_budget


@contextmanager
def isolated_random_state():
    python=random.getstate(); numpy=np.random.get_state(); torch_rng=torch.get_rng_state()
    threads=torch.get_num_threads(); deterministic=torch.are_deterministic_algorithms_enabled()
    try: yield
    finally:
        random.setstate(python); np.random.set_state(numpy); torch.set_rng_state(torch_rng)
        torch.set_num_threads(threads); torch.use_deterministic_algorithms(deterministic)


def evaluation_stages(config,options,start_stage):
    budget=training_budget(config); total=config["training"]["stages"]; first=budget["first_update_stage"]
    every=options["every_stages"]
    if every<1 or options["duration_s"]<=0 or not options["evaluation_seeds"]:
        raise ValueError("Invalid monitoring schedule")
    stages={s for s in range(first,total+1) if s%every==0}|{total}
    if options.get("evaluate_resume_boundary",True): stages.add(start_stage)
    if options.get("evaluate_first_update",True): stages.add(first)
    if options.get("evaluate_first_target_sync",True):
        if config["training"]["target_update_mode"]=="soft": stages.add(first)
        else: stages.add(first+math.ceil(config["training"]["target_update_steps"]/config["training"]["updates_per_stage"])-1)
    return sorted(s for s in stages if start_stage<=s<=total)


class StageMonitor:
    def __init__(self,root: Path,config: dict,options: dict,start_stage: int):
        self.root=root; self.config=deepcopy(config); self.options=options
        self.stages=evaluation_stages(config,options,start_stage); self.rows=[]
        if set(options["evaluation_seeds"])&set(config["evaluation"]["seeds"]):
            raise ValueError("Monitoring seeds must be separate from final test seeds")
        self.eval_config=deepcopy(config)
        self.eval_config["evaluation"]["seeds"]=options["evaluation_seeds"]
        self.eval_config["evaluation"]["duration_s"]=options["duration_s"]
        (root/"monitoring").mkdir(parents=True,exist_ok=True)
        (root/"snapshots").mkdir(exist_ok=True)
        save_json(root/"monitoring/plan.json",dict(options=options,scheduled_stages=self.stages,
            note="monitoring only; original training config and all training random streams are preserved"))

    def progress(self,state,checkpoint,**extra):
        payload=dict(run=str(self.root),state=state,completed_stages=checkpoint["stage"]+1,
            total_stages=self.config["training"]["stages"],steps=checkpoint["steps"],updates=checkpoint["updates"],
            target_syncs=checkpoint.get("target_updates",0),
            timestamp=datetime.now().astimezone().isoformat(),**extra)
        save_json(self.root/"progress.json",payload)
        text=(f"# 正式训练实时状态\n\n运行：`{self.root.name}`\n\n状态：{state}\n\n"
              f"已保存阶段：{payload['completed_stages']}/{payload['total_stages']}；经验：{payload['steps']}；"
              f"更新：{payload['updates']}；目标网络同步：{payload['target_syncs']}。\n\n"
              f"独立评估阶段：{self.stages}；评估seed：{self.options['evaluation_seeds']}。\n\n"
              "完整续训检查点：`train/checkpoint.pt`；阶段模型快照：`snapshots`（仅评估用）。\n\n"
              "阶段评估：`monitoring/stage_*/eval`；实时曲线与对比图：`results/figures/运行名`。\n\n"
              "阶段评估用于诊断，最终测试seed保持独立；单个训练seed及预算完成均不构成收敛证明。\n\n"
              f"更新时间：{payload['timestamp']}\n")
        with atomic_output(ROOT/"reports"/f"运行状态_{self.root.name}.md") as stream: stream.write(text)

    def __call__(self,path,saved):
        stage=saved["stage"]+1
        self.progress("training",saved)
        if stage not in self.stages and stage%self.options["every_stages"]!=0: return
        snapshot=self.root/"snapshots"/f"stage_{stage:03d}.pt"
        keys=("model","scaler_mean","scaler_std","scaler_clip","encoder_version","method","seed","config",
              "capacity_mbps","updates","target_updates","steps","stage","source_hash","checkpoint_schema")
        lean={k:saved[k] for k in keys}; lean["purpose"]="evaluation_only; resume from train/checkpoint.pt"
        if not snapshot.exists():
            with atomic_output(snapshot,binary=True) as stream: torch.save(lean,stream)
        if stage not in self.stages: return
        directory=self.root/"monitoring"/f"stage_{stage:03d}"/"eval"
        if (directory/"metadata.json").exists(): return
        from rl.evaluator import evaluate
        from rl.trainer import save_csv
        self.progress("evaluating",saved)
        with isolated_random_state():
            baseline=self.root/"monitoring/default/eval"
            if not (baseline/"metadata.json").exists():
                evaluate(self.eval_config,"default",saved["seed"],None,baseline)
            results=evaluate(self.eval_config,saved["method"],saved["seed"],snapshot,directory)
            for row in results:
                self.rows.append(dict(stage=stage,training_steps=saved["steps"],gradient_updates=saved["updates"],
                    target_syncs=saved.get("target_updates",0),**row))
            save_csv(self.root/"monitoring/learning_curve.csv",self.rows)
            from scripts.plot_monitoring import plot_monitoring
            from scripts.make_comparisons import build_comparisons
            plot_monitoring(self.root)
            build_comparisons([baseline,directory],ROOT/"results/figures"/self.root.name/f"stage_{stage:03d}",
                              ROOT/"results/summary"/self.root.name/f"stage_{stage:03d}")
        self.progress("training" if stage<self.config["training"]["stages"] else "training_complete",saved)
        with (ROOT/"reports/实验进展.md").open("a",encoding="utf-8") as stream:
            stream.write(f"\n- 自动进展 {datetime.now().astimezone().isoformat()}：`{self.root.name}`阶段{stage}独立评估完成，"
                f"{saved['steps']}经验、{saved['updates']}更新；{len(results)}条评估，CCDF/两方案图/百分比表已保存。尚不判定收敛。\n")
