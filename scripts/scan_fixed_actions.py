"""Stage-E representative fixed-action scan on the calibrated v2 scenarios."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT,load_config,backend_source_hash
from rl.artifacts import atomic_output,save_json
from rl.gates import require_edca_validation
from rl.trainer import save_csv
from sim.backend.base_backend import ControlAction
from sim.backend.ns3_backend import Ns3Backend
from sim.scenarios.resolver import resolve_scenario
from sim.statistics.metrics import aggregate_accounting,summarize

ACS=("vo","vi","be","bk")
ACTIONS={
    "default_p0_beb":ControlAction(profile=0),
    "p1_beb":ControlAction(profile=1),
    "p3_long_txop_beb":ControlAction(profile=3),
    "p6_be_friendly_beb":ControlAction(profile=6),
    "p8_realtime_txop_beb":ControlAction(profile=8),
    "p10_conservative_beb":ControlAction(profile=10),
    "p13_large_cw_beb":ControlAction(profile=13),
    "p15_realtime_priority_beb":ControlAction(profile=15),
    "p0_growth_1p2":ControlAction(profile=0,custom_growth=True,alpha=1,beta1=1.2,beta2=1.2,gamma=7),
    "p0_growth_1p6":ControlAction(profile=0,custom_growth=True,alpha=1,beta1=1.6,beta2=1.6,gamma=7),
    "p0_growth_2p4":ControlAction(profile=0,custom_growth=True,alpha=1,beta1=2.4,beta2=2.4,gamma=7),
    "p3_growth_1p6":ControlAction(profile=3,custom_growth=True,alpha=1,beta1=1.6,beta2=1.6,gamma=7),
}


def run_case(job: tuple) -> dict:
    config,name,scenario,action_name,action,seed,warmup,duration,output=job
    directory=Path(output)/f"{name}__{action_name}__seed{seed}"; directory.mkdir(parents=True,exist_ok=False)
    rows=[]; delays=[[] for _ in ACS]; started=time.monotonic()
    with Ns3Backend(config,scenario,seed,directory,trace=False) as backend:
        for _ in range(round(warmup)): summarize(backend.step(action,1.0),config["network"]["num_sta"],1.0)
        for _ in range(round(duration)):
            events=backend.step(action,1.0); rows.append(summarize(events,config["network"]["num_sta"],1.0))
            for event in events:
                if event.kind=="D": delays[int(event.values[2])].append(event.values[4])
    accounting=aggregate_accounting(rows)
    result={"scenario":name,"action":action_name,"seed":seed,"runtime_s":time.monotonic()-started,
            "goodput_mbps":float(np.mean([r["goodput_mbps"] for r in rows])),
            "external_collision_rate":float(np.mean([r["external_collision_rate"] for r in rows])),**accounting}
    for ac,values in zip(ACS,delays):
        data=np.asarray(values)
        result[f"{ac}_delay_p95_ms"]=float(np.percentile(data,95)) if len(data) else 0.0
        result[f"{ac}_delay_p99_ms"]=float(np.percentile(data,99)) if len(data) else 0.0
        result[f"{ac}_delay_samples"]=len(data)
        if ac in ("vo","vi"):
            deadline=20 if ac=="vo" else 50
            result[f"{ac}_on_time_completion_fraction"]=float(np.sum(data<=deadline)/max(accounting[f"{ac}_at_risk_packets"],1))
    save_csv(directory/"periods.csv",rows); save_json(directory/"summary.json",result)
    with atomic_output(directory/"delay_samples.npz",binary=True) as stream:
        np.savez_compressed(stream,**{ac:np.asarray(v,dtype=np.float64) for ac,v in zip(ACS,delays)})
    return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",default=(641,642)); parser.add_argument("--warmup",type=float,default=10)
    parser.add_argument("--duration",type=float,default=15); parser.add_argument("--actions",nargs="+"); parser.add_argument("--scenarios",nargs="+")
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    gate=require_edca_validation(); config=load_config(ROOT/"configs/default_v2.yaml")
    if gate["source_hash"]!=backend_source_hash(): raise RuntimeError("Backend fingerprint mismatch")
    capacity=json.loads((ROOT/config["traffic"]["capacity_file"]).read_text(encoding="utf-8"))["C_ref_mbps"]
    definitions=load_config(ROOT/config["traffic"]["scenarios"])
    selected_actions=ACTIONS if not args.actions else {k:ACTIONS[k] for k in args.actions}
    selected_scenarios=definitions if not args.scenarios else {k:definitions[k] for k in args.scenarios}
    jobs=[]
    for name,definition in selected_scenarios.items():
        scenario=resolve_scenario(definition,config["network"]["num_sta"],capacity)
        for action_name,action in selected_actions.items():
            for seed in args.seeds: jobs.append((config,name,scenario,action_name,action,seed,args.warmup,args.duration,str(args.output)))
    metadata={"state":"running","started_at":datetime.now().astimezone().isoformat(),"source_hash":gate["source_hash"],
              "config":"configs/default_v2.yaml","accounting_version":3,"seeds":args.seeds,"warmup_s":args.warmup,
              "duration_s":args.duration,"actions":{k:asdict(v) for k,v in selected_actions.items()},"scenarios":list(selected_scenarios),"jobs":len(jobs)}
    save_json(args.output/"metadata.json",metadata); rows=[]; started=time.monotonic()
    with ProcessPoolExecutor(max_workers=config["run"]["workers"]) as pool:
        futures={pool.submit(run_case,job):(job[1],job[3],job[5]) for job in jobs}
        for future in as_completed(futures):
            row=future.result(); rows.append(row); save_csv(args.output/"runs.csv",sorted(rows,key=lambda r:(r["scenario"],r["action"],r["seed"])))
            print(json.dumps({"complete":futures[future],"delivery":row["delivery_fraction"],"vo_p95":row["vo_delay_p95_ms"],"vi_p95":row["vi_delay_p95_ms"]}),flush=True)
    metadata.update(state="complete",runtime_s=time.monotonic()-started,completed_at=datetime.now().astimezone().isoformat())
    save_json(args.output/"metadata.json",metadata)


if __name__=="__main__": main()
