"""Two-STEP audit proving Default fallback and a non-default action reach WifiMac."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config
from rl.tgax_environment import TgaxControlEnvironment


OUT = ROOT / "results/raw/mac_arrival40_gaming160_cmdp_action_gate_20261029"


def main() -> None:
    if OUT.exists(): shutil.rmtree(OUT)
    config=load_config(ROOT / "configs/mac_arrival40_gaming160_cmdp_calibrated.yaml")
    env=TgaxControlEnvironment(config,"compressed_async_age",40,OUT,2.0)
    try:
        env.step(np.asarray(config["action"]["default_action"],dtype=np.int64),1.0,custom=True)
        env.step(np.asarray([8,1,0,2,0],dtype=np.int64),1.0,custom=True)
    finally:
        env.close()
    rows=list(csv.DictReader((OUT/"edca_step_runtime.csv").open(encoding="utf-8")))
    default=[r for r in rows if r["requested_profile"]=="0" and r["requested_custom_growth"]=="0"]
    p8=[r for r in rows if r["requested_profile"]=="8" and r["requested_custom_growth"]=="1"]
    cw=list(csv.DictReader((OUT/"cw_runtime.csv").open(encoding="utf-8")))
    checks={"default_written_to_all_41_nodes_x4_ac":len(default)>=164,
            "p8_custom_written_to_all_41_nodes_x4_ac":len(p8)>=164,
            "p8_edca_differs_from_default":any((r["aifsn"],r["cwmin"],r["cwmax"],r["txop_us"]) not in
                {("2","3","7","2080"),("2","7","15","4096"),("3","15","1023","0"),("7","15","1023","0")}
                for r in p8),
            "cw_trace_present":len(cw)>0}
    result={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,
            "default_rows":len(default),"p8_rows":len(p8),"cw_events":len(cw)}
    (OUT/"action_gate.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    if result["status"]!="PASS": raise RuntimeError(result)


if __name__=="__main__": main()
