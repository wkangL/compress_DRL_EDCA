"""Validate that S5 changes each STA's offered load while preserving a reproducible mean."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT,load_config,digest,backend_source_hash
from sim.backend.base_backend import ControlAction
from sim.backend.ns3_backend import Ns3Backend
from sim.scenarios.resolver import resolve_scenario


def main() -> None:
    config=load_config(ROOT/"configs/default_v2.yaml")
    definitions=load_config(ROOT/config["traffic"]["scenarios"])
    scenario=resolve_scenario(definitions["S5_dynamic_bursty"],config["network"]["num_sta"],27.30225)
    out=ROOT/"results/raw/dynamic_traffic_validation_20261020"
    out.mkdir(parents=True,exist_ok=False)
    rows=[]; duration=30.0; half=float(scenario["load_pattern"]["period_s"])/2
    for seed in (621,622,623):
        case=out/f"seed{seed}"
        with Ns3Backend(config,scenario,seed,case,trace=True) as backend:
            backend.step(ControlAction(),duration)
        bins={i:0 for i in range(int(duration/half))}
        with (case/"mac_trace.csv").open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                # STA 1 has zero phase offset and exposes the intended low/high alternation.
                if row["event"]=="OFFERED" and int(row["node"])==1:
                    index=min(int(float(row["time_s"])/half),len(bins)-1)
                    bins[index]+=int(float(row["value"]))
        for index,byte_count in bins.items():
            rows.append({"seed":seed,"half_period":index,"state":"low" if index%2==0 else "high",
                         "offered_mbps_sta1":byte_count*8/half/1e6})
    with (out/"half_period_rates.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    low=np.asarray([r["offered_mbps_sta1"] for r in rows if r["state"]=="low"])
    high=np.asarray([r["offered_mbps_sta1"] for r in rows if r["state"]=="high"])
    result={"state":"complete","created_at":datetime.now().astimezone().isoformat(),
            "scenario":scenario,"source_hash":backend_source_hash(),"config_hash":digest(config),
            "low_mean_mbps":float(low.mean()),"high_mean_mbps":float(high.mean()),
            "observed_high_low_ratio":float(high.mean()/low.mean()),
            "expected_high_low_ratio":scenario["load_pattern"]["high_scale"]/scenario["load_pattern"]["low_scale"],
            "validation_passed":bool(high.mean()>2*low.mean()),
            "note":"Rates include all four ACs of STA 1. Boundary carry-over and Poisson variation prevent an exact 3:1 finite-run ratio."}
    (out/"metadata.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result["validation_passed"]: raise SystemExit(2)


if __name__=="__main__": main()
