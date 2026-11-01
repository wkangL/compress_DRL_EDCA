"""Baseline load scan for historical scenario compositions under accounting schema 3."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, backend_source_hash, load_config
from rl.artifacts import atomic_output, save_json
from rl.gates import require_edca_validation
from rl.trainer import save_csv
from sim.backend.base_backend import ControlAction
from sim.backend.ns3_backend import Ns3Backend
from sim.scenarios.resolver import resolve_scenario
from sim.statistics.metrics import aggregate_accounting, summarize


ACS=("vo","vi","be","bk")
DEFAULT_SCALES=(0.40,0.55,0.70,0.85,1.00,1.15)
DEFAULT_SEEDS=(601,602)


def token(value: float) -> str:
    return f"{value:.2f}".replace(".","p")


def busy_fraction(events, start: float, end: float) -> float:
    spans=[]
    for event in events:
        if event.kind=="B":
            left=max(start,event.values[0]); right=min(end,event.values[0]+event.values[1])
            if right>left: spans.append((left,right))
    spans.sort(); covered=0.0; last=start
    for left,right in spans:
        covered += max(0.0,right-max(left,last)); last=max(last,right)
    return covered/max(end-start,1e-12)


def run_case(job: tuple[dict,str,dict,float,int,str,float,float]) -> dict:
    config,name,definition,scale,seed,output,warmup,measure=job
    capacity=json.loads((ROOT/config["traffic"]["capacity_file"]).read_text(encoding="utf-8"))["C_ref_mbps"]
    base=resolve_scenario(definition,config["network"]["num_sta"],capacity)
    base_rates=base.get("per_sta_mbps_by_ac",[base["offered_mbps"]*x/config["network"]["num_sta"] for x in base["ac_mix"]])
    rates=[float(x)*scale for x in base_rates]; total=sum(rates)
    scenario=dict(offered_mbps=total*config["network"]["num_sta"],ac_mix=[x/total for x in rates],
                  ul_fraction=base["ul_fraction"],arrival="poisson")
    directory=Path(output)/f"{name}_scale{token(scale)}_seed{seed}"
    directory.mkdir(parents=True,exist_ok=False)
    rows=[]; delays={f"{d}_{ac}":[] for d in ("ul","dl") for ac in ACS}; queues={f"{d}_{ac}":[] for d in ("ul","dl") for ac in ACS}
    action=ControlAction(); started=time.monotonic()
    with Ns3Backend(config,scenario,seed,directory,trace=False) as backend:
        if backend.accounting_version<3: raise RuntimeError("Load scan requires accounting schema 3")
        for _ in range(round(warmup)):
            summarize(backend.step(action,1.0),config["network"]["num_sta"],1.0)
        for second in range(round(measure)):
            start=backend.time; events=backend.step(action,1.0); metric=summarize(events,config["network"]["num_sta"],1.0)
            rows.append(dict(second=second+1,busy_fraction=busy_fraction(events,start,backend.time),**metric))
            for event in events:
                if event.kind=="D":
                    node=int(event.values[1]); ac=int(event.values[2]); delays[f"{'dl' if node==0 else 'ul'}_{ACS[ac]}"].append(event.values[4])
                elif event.kind=="P":
                    node=int(event.values[1]); ac=int(event.values[2]); queues[f"{'dl' if node==0 else 'ul'}_{ACS[ac]}"].append(int(event.values[10]))
    accounting=aggregate_accounting(rows)
    result=dict(scenario=name,scale=scale,seed=seed,per_sta_mbps_by_ac=json.dumps(rates),
        per_sta_total_mbps=total,network_target_mbps=scenario["offered_mbps"],ul_fraction=scenario["ul_fraction"],
        warmup_s=warmup,measure_s=measure,runtime_s=time.monotonic()-started,
        goodput_mbps=np.mean([r["goodput_mbps"] for r in rows]),busy_fraction=np.mean([r["busy_fraction"] for r in rows]),
        external_collision_rate=np.mean([r["external_collision_rate"] for r in rows]),
        combined_collision_event_fraction=np.mean([r["combined_collision_event_fraction"] for r in rows]),
        ack_cts_timeout_ratio=np.mean([r["ack_cts_timeout_ratio"] for r in rows]),
        delivery_fraction=accounting["delivery_fraction"],final_failure_fraction=accounting["final_failure_fraction"],
        unfinished_fraction=accounting["unfinished_fraction"],queue_rejection_fraction=accounting["queue_rejection_fraction_of_generated"],
        attempts=sum(r["attempts"] for r in rows),timeouts=sum(r["timeouts"] for r in rows))
    for ac in ACS:
        samples=delays[f"ul_{ac}"]+delays[f"dl_{ac}"]
        result[f"{ac}_goodput_mbps"]=np.mean([r[f"{ac}_goodput_mbps"] for r in rows])
        result[f"{ac}_delivery_fraction"]=accounting[f"{ac}_delivery_fraction"]
        result[f"{ac}_final_failure_fraction"]=accounting[f"{ac}_final_failure_fraction"]
        result[f"{ac}_unfinished_fraction"]=accounting[f"{ac}_unfinished_fraction"]
        result[f"{ac}_delay_valid"]=bool(samples)
        result[f"{ac}_delay_p95_ms"]=float(np.percentile(samples,95)) if samples else 0.0
        result[f"{ac}_delay_p99_ms"]=float(np.percentile(samples,99)) if samples else 0.0
        result[f"{ac}_delay_samples"]=len(samples)
        for direction in ("ul","dl"):
            q=queues[f"{direction}_{ac}"]; ds=delays[f"{direction}_{ac}"]
            result[f"{direction}_{ac}_queue_mean"]=float(np.mean(q)) if q else 0.0
            result[f"{direction}_{ac}_queue_p95"]=float(np.percentile(q,95)) if q else 0.0
            result[f"{direction}_{ac}_queue_max"]=max(q,default=0)
            result[f"{direction}_{ac}_queue_nonempty_fraction"]=float(np.mean(np.asarray(q)>0)) if q else 0.0
            result[f"{direction}_{ac}_delay_valid"]=bool(ds)
            result[f"{direction}_{ac}_delay_p95_ms"]=float(np.percentile(ds,95)) if ds else 0.0
    save_csv(directory/"periods.csv",rows); save_json(directory/"summary.json",result); save_json(directory/"scenario.json",scenario)
    with atomic_output(directory/"delay_samples.npz",binary=True) as stream:
        np.savez_compressed(stream,**{key:np.asarray(value,dtype=np.float64) for key,value in delays.items()})
    return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--scales",type=float,nargs="+",default=DEFAULT_SCALES)
    parser.add_argument("--seeds",type=int,nargs="+",default=DEFAULT_SEEDS)
    parser.add_argument("--scenarios",nargs="+",help="Optional exact scenario names; default scans all")
    parser.add_argument("--warmup",type=float,default=10.0); parser.add_argument("--measure",type=float,default=20.0)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    gate=require_edca_validation(); config=load_config(ROOT/"configs/full.yaml")
    if gate["source_hash"]!=backend_source_hash(): raise RuntimeError("Backend fingerprint mismatch")
    scenarios=load_config(ROOT/config["traffic"]["scenarios"])
    if args.scenarios:
        unknown=set(args.scenarios)-set(scenarios)
        if unknown: raise ValueError(f"Unknown scenarios: {sorted(unknown)}")
        scenarios={name:scenarios[name] for name in args.scenarios}
    jobs=[(config,name,definition,scale,seed,str(args.output),args.warmup,args.measure)
          for name,definition in scenarios.items() for scale in args.scales for seed in args.seeds]
    metadata=dict(state="running",source_hash=gate["source_hash"],accounting_version=3,scenarios=list(scenarios),
        scales=args.scales,seeds=args.seeds,warmup_s=args.warmup,measure_s=args.measure,jobs=len(jobs),
        retry_limit=config["network"]["retry_limit"],queue_limits=config["queues"]["limits_packets"],
        queue_lifetime_s=config["queues"]["max_delay_s"],started_at=datetime.now().astimezone().isoformat())
    save_json(args.output/"metadata.json",metadata); rows=[]; started=time.monotonic()
    with ProcessPoolExecutor(max_workers=config["run"]["workers"]) as pool:
        futures={pool.submit(run_case,job):(job[1],job[3],job[4]) for job in jobs}
        for future in as_completed(futures):
            row=future.result(); rows.append(row); save_csv(args.output/"runs.csv",sorted(rows,key=lambda r:(r["scenario"],r["scale"],r["seed"])))
            print(json.dumps({"complete":futures[future],"delivery":row["delivery_fraction"],"vo_p95":row["vo_delay_p95_ms"],"vi_p95":row["vi_delay_p95_ms"]}),flush=True)
    metadata.update(state="complete",runtime_s=time.monotonic()-started,completed_at=datetime.now().astimezone().isoformat())
    save_json(args.output/"metadata.json",metadata); (args.output/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")


if __name__=="__main__": main()
