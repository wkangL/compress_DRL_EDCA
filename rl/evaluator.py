from __future__ import annotations
from datetime import datetime
from pathlib import Path
from copy import deepcopy
import json,time
import numpy as np
import torch
from project_config import ROOT,load_config
from reporting.sta_state import StateScaler
from models.branching_ddqn import ControlNetwork
from sim.statistics.ap_state import GLOBAL_DIM
from rl.environment import ControlEnvironment
from rl.gates import require_edca_validation
from rl.trainer import InferenceCache,set_seed,save_csv
from rl.artifacts import atomic_output,save_json
from sim.statistics.metrics import aggregate_accounting
from sim.scenarios.resolver import resolve_scenario


def evaluate(config: dict, method: str, train_seed: int, checkpoint: Path|None, directory: Path) -> list[dict]:
    gate=require_edca_validation(); started=time.monotonic()
    directory.mkdir(parents=True,exist_ok=False)
    set_seed(train_seed,config["training"]["threads"])
    capacity_record=json.loads((ROOT/config["traffic"]["capacity_file"]).read_text(encoding="utf-8"))
    if capacity_record["source_hash"]!=gate["source_hash"]:
        raise ValueError("Evaluation capacity calibration does not match backend")
    capacity=capacity_record["C_ref_mbps"]
    model=None; scaler=None; version=0
    if method!="default":
        if checkpoint is None: raise ValueError("A trained checkpoint is required")
        saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
        expected=deepcopy(saved["config"]); expected["evaluation"]=deepcopy(config["evaluation"])
        expected["run"]["workers"]=config["run"]["workers"]
        if (saved.get("checkpoint_schema",0)<3 or expected!=config or
            saved["source_hash"]!=gate["source_hash"] or saved["method"]!=method or saved["seed"]!=train_seed):
            raise ValueError("Checkpoint backend/method mismatch")
        model=ControlNetwork(config,method,GLOBAL_DIM); model.load_state_dict(saved["model"]); model.eval()
        scaler=StateScaler(saved["scaler_clip"]); scaler.mean=saved["scaler_mean"]; scaler.std=saved["scaler_std"]; scaler.fitted=True
        version=saved["encoder_version"]
        del saved
    rows=[]; summary=[]
    scenarios=load_config(ROOT/config["traffic"]["scenarios"])
    for name,s in scenarios.items():
        scenario=resolve_scenario(s,config["network"]["num_sta"],capacity)
        for eval_seed in config["evaluation"]["seeds"]:
            env=ControlEnvironment(config,method,scenario,eval_seed,directory/f"{name}_seed{eval_seed}",capacity,version)
            inference=InferenceCache(model,scaler) if model is not None else None
            observation=env.reset_observation(); episode_rows=[]; delays=[[] for _ in range(4)]
            station_bytes=np.zeros(config["network"]["num_sta"])
            duration=config["evaluation"]["duration_s"]; period=config["timing"]["decision_s"]
            try:
                warmup=config["evaluation"].get("traffic_warmup_s",0.)
                warmup_end=env.time+warmup
                while env.time<warmup_end-1e-9:
                    action=inference.select(observation) if inference else env.action
                    observation,_,_=env.step(action,min(period,warmup_end-env.time))
                measurement_start=env.time; measurement_end=measurement_start+duration
                while env.time<measurement_end-1e-9:
                    action=inference.select(observation) if inference else env.action
                    observation,_,metrics=env.step(action,min(period,measurement_end-env.time))
                    for e in env.last_events:
                        if e.kind=="D":
                            delays[int(e.values[2])].append(e.values[4])
                            station_bytes[int(e.values[1])]+=e.values[3]
                    row=dict(method=method,train_seed=train_seed,eval_seed=eval_seed,scenario=name,
                             time_s=env.time-measurement_start,
                             action=json.dumps(action.tolist()),**metrics)
                    rows.append(row); episode_rows.append(row)
            finally: env.close()
            # Exact successful-packet delay samples, never averaged period percentiles.
            with atomic_output(directory/f"{name}_seed{eval_seed}"/"delay_samples.npz",binary=True) as stream:
                np.savez_compressed(stream,**{ac:np.asarray(samples,dtype=np.float64)
                    for ac,samples in zip(("vo","vi","be","bk"),delays)})
            aggregate={k:float(np.mean([r[k] for r in episode_rows])) for k in metrics if isinstance(metrics[k],(int,float,bool,np.number))}
            aggregate.update(method=method,train_seed=train_seed,eval_seed=eval_seed,scenario=name)
            for ac,samples in zip(("vo","vi","be","bk"),delays):
                aggregate[f"{ac}_delay_p95_ms"]=float(np.percentile(samples,95)) if samples else 0.
                aggregate[f"{ac}_delay_mean_ms"]=float(np.mean(samples)) if samples else 0.
                aggregate[f"{ac}_delay_valid"]=bool(samples)
            total=station_bytes.sum()
            aggregate["jain_fairness"]=float(total**2/(len(station_bytes)*np.square(station_bytes).sum())) if total else 1.
            for rate,num,den in [("collision_rate","collision_count","air_attempts"),
                                  ("ack_cts_timeout_ratio","timeouts","attempts")]:
                aggregate[rate]=sum(r[num] for r in episode_rows)/max(sum(r[den] for r in episode_rows),1)
            aggregate["drop_rate"]=sum(r["failed_count"]+r["discarded_count"] for r in episode_rows)/max(sum(r["offered_packets"] for r in episode_rows),1)
            aggregate.update(aggregate_accounting(episode_rows))
            summary.append(aggregate)
            save_csv(directory/"periods.csv",rows); save_csv(directory/"evaluation.csv",summary)
    (directory/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    save_json(directory/"metadata.json",dict(method=method,seed=train_seed,checkpoint=str(checkpoint),
        runtime_s=time.monotonic()-started,source_hash=gate["source_hash"],completed_at=datetime.now().astimezone().isoformat(),
        policy="greedy; no exploration",traffic_warmup_s=config["evaluation"].get("traffic_warmup_s",0.),
        delay_definition="MAC enqueue to successful ACK; measurement-window departures only; successful packets only",
        accounting_version=summary[0].get("accounting_version",1),
        count_definition="episode totals; pending counts are start/end snapshots"))
    return summary
