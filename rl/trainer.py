"""Central staged TD optimization: no optimizer step occurs during an episode."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import csv,json,random,subprocess,time
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from project_config import ROOT,load_config
from models.branching_ddqn import ControlNetwork
from reporting.sta_state import StateScaler
from rl.environment import ControlEnvironment
from rl.gates import require_edca_validation
from rl.replay_buffer import ReplayBuffer,Transition
from rl.cmdp_reward import lagrangian_reward
from rl.artifacts import atomic_output,save_json
from rl.training_budget import training_budget
from sim.statistics.ap_state import GLOBAL_DIM
from sim.scenarios.resolver import resolve_scenario


def set_seed(seed: int, threads: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def tensors(states: list[dict], scaler: StateScaler) -> tuple[torch.Tensor,...]:
    histories=scaler.transform(np.stack([s["history"] for s in states]))
    return (torch.from_numpy(histories),*(torch.tensor(np.stack([s[k] for s in states]),dtype=torch.float32)
                                         for k in ("age","valid","global_stats","action")))


def explore(model: ControlNetwork, rng: np.random.Generator) -> np.ndarray:
    result=np.asarray([rng.integers(n) for n in model.branches],dtype=np.int64)
    pairs=torch.nonzero(model.beta_mask).numpy()
    result[model.scalar_offset+1:model.scalar_offset+3]=pairs[rng.integers(len(pairs))]
    if (model.default_fallback_index is not None and
            np.all(result[:model.profile_branch_count]==model.default_fallback_index)):
        result[model.scalar_offset:]=0
    return result


def freeze_target(target: ControlNetwork) -> ControlNetwork:
    """Target inference is independent, eval-only and never accumulates gradients."""
    target.eval(); target.requires_grad_(False)
    return target


@torch.no_grad()
def target_distance(model: ControlNetwork, target: ControlNetwork) -> float:
    total=0.; count=0
    for online, lagged in zip(model.parameters(),target.parameters()):
        total+=float(torch.square(online-lagged).sum()); count+=online.numel()
    return float((total/max(count,1))**.5)


@torch.no_grad()
def update_target(model: ControlNetwork, target: ControlNetwork, update: int, config: dict) -> tuple[bool,float,float]:
    before=target_distance(model,target); mode=config["target_update_mode"]
    if mode=="hard":
        changed=update%config["target_update_steps"]==0
        if changed: target.load_state_dict(model.state_dict())
    else:
        changed=True; tau=config["target_tau"]
        online=model.state_dict(); lagged=target.state_dict()
        for name,value in lagged.items():
            value.copy_(value.lerp(online[name],tau) if value.is_floating_point() else online[name])
    freeze_target(target)
    return changed,before,target_distance(model,target)


def replay_audit(replay: ReplayBuffer, stage: int) -> dict:
    scenarios=Counter(t.scenario for t in replay.data)
    branch_count=(len(replay.data[0].action) if replay.data else 0)
    branches=[Counter(int(t.action[i]) for t in replay.data) for i in range(branch_count)]
    return dict(stage=stage,transitions=len(replay),scenario_counts=json.dumps(dict(sorted(scenarios.items()))),
                action_branch_counts=json.dumps([{str(k):v for k,v in sorted(c.items())} for c in branches]),
                exploratory_fraction=float(np.mean([t.exploratory for t in replay.data])) if replay.data else 0.,
                terminal_count=sum(t.done for t in replay.data),truncated_count=sum(t.truncated for t in replay.data))


class InferenceCache:
    """A frozen episode encoder evaluates only new received snapshots; fresh oracles encode on demand."""
    def __init__(self, model: ControlNetwork, scaler: StateScaler):
        self.model=model; self.scaler=scaler; self.times=None; self.latent=None

    @torch.no_grad()
    def select(self, state: dict) -> np.ndarray:
        args=tensors([state],self.scaler)
        if not self.model.compressed:
            return self.model.greedy(self.model(*args))[0].numpy()
        times=state["report_time"]
        if self.times is None:
            self.times=np.full(len(times),-np.inf)
            self.latent=torch.zeros(1,len(times),4,self.model.encoder.output.out_features)
        fresh=self.model.method=="compressed_fresh"
        changed=((times!=self.times)|fresh)&(state["valid"]>0)
        if changed.any():
            self.latent[:,changed]=self.model.encoder(args[0][:,changed])
            self.times[changed]=times[changed]
        return self.model.greedy(self.model(*args,latent=self.latent))[0].numpy()


def td_update(model: ControlNetwork,target: ControlNetwork,batch: list[Transition],scaler: StateScaler,
              optimizer: torch.optim.Optimizer,config: dict,lagrange: dict|None=None) -> dict:
    state=tensors([t.state for t in batch],scaler); following=tensors([t.next_state for t in batch],scaler)
    actions=torch.tensor(np.stack([t.action for t in batch]),dtype=torch.int64)
    if lagrange is None:
        reward_values=[t.reward for t in batch]
    else:
        if any(t.utility is None or t.costs is None for t in batch):
            raise ValueError("CMDP update sampled a transition without raw utility/costs")
        reward_values=[lagrangian_reward(t.utility,t.costs,lagrange["budgets"],lagrange["multipliers"],
                                         excess_only=lagrange.get("excess_only",False))
                       for t in batch]
    rewards=torch.tensor(reward_values,dtype=torch.float32)
    terminal=torch.tensor([t.done for t in batch],dtype=torch.float32)
    qs=model(*state)
    selected=torch.stack([q.gather(1,actions[:,i,None]).squeeze(1) for i,q in enumerate(qs)],dim=1)
    with torch.no_grad():
        greedy=model.greedy(model(*following))
        target_q=target(*following)
        target_selected=torch.stack([q.gather(1,greedy[:,i,None]).squeeze(1) for i,q in enumerate(target_q)],1)
        next_value=target_selected.mean(1)
        expected=rewards+config["discount"]*(1-terminal)*next_value
    loss=F.smooth_l1_loss(selected,expected[:,None].expand_as(selected))
    optimizer.zero_grad(set_to_none=True); loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config["grad_clip"])
    if not torch.isfinite(loss) or not torch.isfinite(norm): raise FloatingPointError("Nonfinite TD training")
    optimizer.step()
    with torch.no_grad():
        z=model.encoder(state[0]); variance=z.flatten(0,2).var(0,unbiased=False)
    with torch.no_grad():
        error=expected[:,None]-selected
        def stats(prefix,x):
            return {f"{prefix}_mean":float(x.mean()),f"{prefix}_min":float(x.min()),
                    f"{prefix}_max":float(x.max()),f"{prefix}_std":float(x.std(unbiased=False))}
        result={"td_loss":float(loss.detach()),**stats("q",selected.detach()),
                **stats("target_q",target_selected),**stats("td_target",expected),
                **stats("td_error",error),"td_error_abs_mean":float(error.abs().mean()),
                "td_error_max_abs":float(error.abs().max()),"reward_mean":float(rewards.mean()),
                "bootstrap_fraction":float((1-terminal).mean()),
                "terminal_fraction":float(terminal.mean()),
                "truncated_fraction":float(np.mean([t.truncated for t in batch])),
                "grad_norm":float(norm),"latent_variance":variance.tolist()}
    return result


def save_csv(path: Path,rows: list[dict]) -> None:
    if not rows: return
    with atomic_output(path) as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def resume_logs(saved: dict, directory: Path) -> tuple[list,list,list]:
    """The checkpoint is the commit boundary, even if later episodes wrote CSVs."""
    names=("steps.csv","episodes.csv","losses.csv")
    result=[]
    for name in names:
        if "training_logs" in saved:
            rows=deepcopy(saved["training_logs"][name])
        else:
            path=directory/name
            rows=[]
            if path.exists():
                with path.open(encoding="utf-8") as stream:
                    rows=[row for row in csv.DictReader(stream) if int(row["stage"])<=saved["stage"]]
        result.append(rows)
    expected=(saved["steps"],(saved["stage"]+1)*saved["config"]["training"]["episodes_per_stage"],saved["updates"])
    if tuple(map(len,result))!=expected:
        raise ValueError("Checkpoint-boundary training logs are missing or incomplete; cannot resume exactly")
    return tuple(result)


def train(config: dict, method: str, seed: int, directory: Path, resume: Path|None=None,
          stop_after_stage: int|None=None, on_stage=None) -> Path:
    budget=training_budget(config)
    if stop_after_stage is not None and not 1<=stop_after_stage<=config["training"]["stages"]:
        raise ValueError("stop_after_stage must be within the configured stages (one-based)")
    gate=require_edca_validation(); directory.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); started_at=datetime.now().astimezone().isoformat()
    (directory/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    save_json(directory/"budget.json",budget)
    save_json(directory/"status.json",dict(state="initializing",total_stages=config["training"]["stages"],
              resume_checkpoint=str(resume.resolve()) if resume is not None else None))
    capacity_record=json.loads((ROOT/config["traffic"]["capacity_file"]).read_text(encoding="utf-8"))
    if capacity_record["source_hash"]!=gate["source_hash"]: raise RuntimeError("Recalibrate capacity for this backend")
    capacity=capacity_record["C_ref_mbps"]
    scenarios=load_config(ROOT/config["traffic"]["scenarios"])
    tc=config["training"]; set_seed(seed,tc["threads"]); rng=np.random.default_rng(seed)
    model=ControlNetwork(config,method,GLOBAL_DIM); target=freeze_target(deepcopy(model))
    optimizer=torch.optim.AdamW([{"params":model.q.parameters(),"lr":tc["q_lr"]},
                                {"params":list(model.encoder.parameters())+list(model.aggregator.parameters()),"lr":tc["representation_lr"]}])
    scaler=StateScaler(config["state"]["delay_clip_ms"])
    replay=ReplayBuffer(tc["replay_capacity"],seed)
    logs=[]; episodes=[]; losses=[]; replay_audits=[]; steps=updates=target_updates=version=0
    first_stage=0
    if resume is None:
        # Fit only on default-policy training rollouts, using all four training scenarios.
        calibration=[]
        for index,(name,s) in enumerate(scenarios.items()):
            scenario=resolve_scenario(s,config["network"]["num_sta"],capacity)
            env=ControlEnvironment(config,"default",scenario,seed*100000+9000+index,directory/f"scaler_{name}",capacity,0)
            try:
                warmup=tc.get("traffic_warmup_s",0.)
                for _ in range(round(warmup/config["timing"]["decision_s"])):
                    env.step(env.action,config["timing"]["decision_s"])
                for _ in range(round(tc["episode_s"]/config["timing"]["decision_s"])):
                    obs,_,_=env.step(env.action,config["timing"]["decision_s"])
                    calibration.append(env.history.all())
            finally: env.close()
        scaler.fit(np.stack(calibration)); del calibration
    else:
        saved=torch.load(resume,map_location="cpu",weights_only=False)
        if "replay" not in saved:
            raise ValueError("This early checkpoint has no replay/RNG state; exact continuation is unavailable")
        expected=deepcopy(saved["config"]); expected["training"]["stages"]=tc["stages"]
        expected["run"]["workers"]=config["run"]["workers"]
        if expected!=config or saved["source_hash"]!=gate["source_hash"] or saved["method"]!=method or saved["seed"]!=seed:
            raise ValueError("Resume requires identical research config, method, seed and backend")
        model.load_state_dict(saved["model"]); target.load_state_dict(saved["target"]); freeze_target(target); optimizer.load_state_dict(saved["optimizer"])
        scaler.mean=saved["scaler_mean"]; scaler.std=saved["scaler_std"]; scaler.fitted=True
        replay.data.extend(saved["replay"]); replay.rng.bit_generator.state=saved["replay_rng"]
        rng.bit_generator.state=saved["rng"]; torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"]); random.setstate(saved["python_rng"])
        steps=saved["steps"]; updates=saved["updates"]; target_updates=saved["target_updates"]
        version=saved["encoder_version"]; first_stage=saved["stage"]+1
        if first_stage>=tc["stages"]: raise ValueError("No remaining stage; increase stages in a separate config")
        if stop_after_stage is not None and stop_after_stage<=first_stage:
            raise ValueError("stop_after_stage must be later than the saved checkpoint")
        logs,episodes,losses=resume_logs(saved,resume.parent)
        replay_audits=deepcopy(saved["training_logs"].get("replay_audit.csv",[]))
        save_json(directory/"resumed_from.json",dict(checkpoint=str(resume.resolve()),first_stage=first_stage))
        if on_stage is not None:
            on_stage(resume,saved)
    for stage in range(first_stage,tc["stages"]):
        model.eval()
        frozen={k:v.detach().clone() for k,v in model.encoder.state_dict().items()}
        for episode in range(tc["episodes_per_stage"]):
            name=list(scenarios)[int(rng.integers(len(scenarios)))]; s=scenarios[name]
            scenario=resolve_scenario(s,config["network"]["num_sta"],capacity)
            env=ControlEnvironment(config,method,scenario,seed*100000+stage*100+episode+1,
                                   directory/f"stage{stage}_episode{episode}",capacity,version)
            inference=InferenceCache(model,scaler); obs=env.reset_observation(); total=0.
            count=round(tc["episode_s"]/config["timing"]["decision_s"])
            try:
                warmup=tc.get("traffic_warmup_s",0.)
                warmup_count=round(warmup/config["timing"]["decision_s"])
                if abs(warmup_count*config["timing"]["decision_s"]-warmup)>1e-9:
                    raise ValueError("training traffic_warmup_s must be a whole number of decision periods")
                for _ in range(warmup_count):
                    action=inference.select(obs)
                    obs,_,_=env.step(action,config["timing"]["decision_s"])
                for step in range(count):
                    eps=tc["epsilon_end"]+(tc["epsilon_start"]-tc["epsilon_end"])*max(0,1-steps/tc["epsilon_steps"])
                    exploratory=bool(rng.random()<eps)
                    action=explore(model,rng) if exploratory else inference.select(obs)
                    following,r,metrics=env.step(action,config["timing"]["decision_s"])
                    truncated=step==count-1
                    replay.add(Transition(obs,action,r,following,False,version,truncated,name,exploratory)); obs=following
                    total+=r; steps+=1
                    logs.append(dict(stage=stage,episode=episode,step=steps,scenario=name,epsilon=eps,
                                     exploratory=exploratory,terminated=False,truncated=truncated,
                                     encoder_version=version,action=json.dumps(action.tolist()),**metrics))
                for k,v in model.encoder.state_dict().items():
                    if not torch.equal(v,frozen[k]): raise RuntimeError("Encoder changed during an episode")
            finally: env.close()
            episodes.append(dict(stage=stage,episode=episode,scenario=name,reward=total/count,encoder_version=version))
            save_csv(directory/"episodes.csv",episodes); save_csv(directory/"steps.csv",logs)
        if len(replay)>=max(tc["warmup"],tc["batch_size"]):
            model.train()
            for _ in range(tc["updates_per_stage"]):
                stats=td_update(model,target,replay.sample(tc["batch_size"]),scaler,optimizer,tc); updates+=1
                synced,distance_before,distance_after=update_target(model,target,updates,tc)
                target_updates+=int(synced)
                stats.update(target_sync=synced,target_update_mode=tc["target_update_mode"],
                             target_distance_before=distance_before,target_distance_after=distance_after)
                losses.append(dict(stage=stage,update=updates,**{k:json.dumps(v) if isinstance(v,list) else v for k,v in stats.items()}))
            # All subsequent episodes instantiate empty caches and use this unified version.
            version+=1
        save_csv(directory/"losses.csv",losses)
        replay_audits.append(replay_audit(replay,stage)); save_csv(directory/"replay_audit.csv",replay_audits)
        checkpoint=dict(model=model.state_dict(),target=target.state_dict(),optimizer=optimizer.state_dict(),
                        scaler_mean=scaler.mean,scaler_std=scaler.std,scaler_clip=scaler.delay_clip_ms,
                        encoder_version=version,method=method,seed=seed,config=config,capacity_mbps=capacity,
                        updates=updates,target_updates=target_updates,steps=steps,stage=stage,source_hash=gate["source_hash"],
                        replay=list(replay.data),replay_rng=replay.rng.bit_generator.state,rng=rng.bit_generator.state,
                        torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                        checkpoint_schema=3,training_logs={"steps.csv":logs,"episodes.csv":episodes,
                                                          "losses.csv":losses,"replay_audit.csv":replay_audits})
        with atomic_output(directory/"checkpoint.pt",binary=True) as stream:
            torch.save(checkpoint,stream)
        if on_stage is not None:
            on_stage(directory/"checkpoint.pt",checkpoint)
        complete=stage+1==tc["stages"]
        stopped=stop_after_stage is not None and stage+1>=stop_after_stage
        save_json(directory/"status.json",dict(state="complete" if complete else "paused" if stopped else "running",
                  completed_stages=stage+1,total_stages=tc["stages"],steps=steps,updates=updates,
                  target_updates=target_updates,target_update_mode=tc["target_update_mode"],
                  checkpoint=str((directory/"checkpoint.pt").resolve())))
        print(f"{method} seed={seed} stage={stage+1}/{tc['stages']} replay={len(replay)} updates={updates}",flush=True)
        if stopped: break
    metadata=dict(seed=seed,method=method,start_time=started_at,runtime_s=time.monotonic()-started,updates=updates,
                  target_updates=target_updates,target_update_mode=tc["target_update_mode"],
                  encoder_version=version,ns3_commit="43dce6710b8df69685e3479c1d33a571dda714ea",git_commit=None,
                  state="complete" if stage+1==tc["stages"] else "paused",completed_stages=stage+1,
                  note="Quick is a pipeline check; convergence is assessed separately.")
    save_json(directory/"metadata.json",metadata)
    return directory/"checkpoint.pt"
