"""One real low-load ns-3 run, without AI."""
from pathlib import Path
import sys, json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT,load_config
from sim.backend.ns3_backend import Ns3Backend
from sim.backend.base_backend import ControlAction
from sim.statistics.metrics import summarize

if __name__=="__main__":
    c=load_config(ROOT/"configs/quick.yaml"); c["network"]["num_sta"]=1
    s={"offered_mbps":1.,"ac_mix":[0,0,1,0],"ul_fraction":1.}
    with Ns3Backend(c,s,11,ROOT/"results/raw/backend_smoke",trace=True) as b:
        events=b.step(ControlAction(),2.)
    result=summarize(events,1,2.)
    print(json.dumps(result,indent=2))
