"""Estimate C_ref from high-load default-EDCA measurements, never from the old 25.54."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT,load_config
from experiments.run_edca_validation import run_case
from sim.backend.base_backend import ControlAction
from rl.gates import require_edca_validation


def main() -> None:
    gate=require_edca_validation(); c=load_config(ROOT/"configs/quick.yaml")
    definition=load_config(ROOT/"configs/capacity.yaml")
    root=ROOT/"results/raw"/datetime.now().strftime("capacity_%Y%m%d_%H%M%S"); root.mkdir(parents=True)
    jobs=[]
    c["validation"]["duration_s"]=definition["duration_s"]
    for load in definition["offered_mbps"]:
        for seed in definition["seeds"]:
            scenario={"offered_mbps":load,"ac_mix":[0,0,1,0],"ul_fraction":1.}
            jobs.append((f"load_{load}",c,scenario,seed,ControlAction(),str(root)))
    with ProcessPoolExecutor(max_workers=c["run"]["workers"]) as pool: rows=list(pool.map(run_case,jobs))
    if any(r["status"]!="measured" for r in rows): raise RuntimeError("Capacity run failed")
    means=[np.mean([r["goodput_mbps"] for r in rows if r["case"]==f"load_{load}"]) for load in definition["offered_mbps"]]
    spread=(max(means)-min(means))/np.mean(means)
    if spread>definition["plateau_relative_tolerance"]:
        raise RuntimeError(f"High-load tests did not establish a capacity plateau: {means}")
    result={"C_ref_mbps":float(np.mean(means)),"load_means":list(map(float,means)),"definition":definition,
            "raw_directory":str(root),"source_hash":gate["source_hash"],"rows":rows}
    (ROOT/"results/summary/capacity.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__=="__main__": main()
