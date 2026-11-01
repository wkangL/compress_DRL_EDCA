"""Read-only dependency/source inspection for reproducibility."""
from __future__ import annotations
from pathlib import Path
import importlib.metadata,json,platform,sys,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT,load_config
from sim.backend.ns3_backend import find_binary


def main() -> None:
    packages={}
    for name in ("torch","numpy","matplotlib","scipy","PyYAML","pytest"):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    config=load_config(ROOT/"configs/quick.yaml")
    result=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=packages,
                ns3_binary=str(find_binary(config)),ns3_source=str(ROOT/config["backend"]["source"]))
    result["ns3_commit"]=subprocess.check_output(["git","-C",result["ns3_source"],"rev-parse","HEAD"],text=True).strip()
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
