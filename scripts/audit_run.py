"""Read-only completeness checks before publishing experiment summaries."""
from pathlib import Path
import argparse
import csv
import json
import math
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT,load_config


def audit_run(root: Path, require_full_matrix: bool = False) -> dict:
    config=json.loads((root/"config.json").read_text(encoding="utf-8"))
    status=json.loads((root/"status.json").read_text(encoding="utf-8"))
    if status.get("state")!="complete":
        raise ValueError("Run is incomplete or has failed seeds")
    jobs=status.get("jobs",[])
    keys=[(j["method"],j["seed"]) for j in jobs]
    if not keys or len(keys)!=len(set(keys)) or any(j.get("status")!="complete" for j in jobs):
        raise ValueError("Missing, duplicated or unsuccessful jobs")
    if status.get("planned_jobs") is not None:
        planned={(j["method"],j["seed"]) for j in status["planned_jobs"]}
        if set(keys)!=planned:
            raise ValueError("Completed jobs differ from the planned matrix")
    if require_full_matrix:
        expected={(m,s) for m in config["evaluation"]["methods"]
                  for s in (config["run"]["seeds"][:1] if m=="default" else config["run"]["seeds"])}
        if set(keys)!=expected:
            raise ValueError("Run does not contain the complete configured method/seed matrix")
    scenarios=load_config(ROOT/config["traffic"]["scenarios"])
    expected_rows={(s,int(seed)) for s in scenarios for seed in config["evaluation"]["seeds"]}
    count=0
    expected_dirs={f"{m}_seed{s}" for m,s in keys}
    actual_dirs={p.parent.parent.name for p in root.glob("*_seed*/eval/evaluation.csv")}
    if actual_dirs!=expected_dirs:
        raise ValueError("Evaluation directories differ from completed jobs")
    for job in jobs:
        path=root/f"{job['method']}_seed{job['seed']}"/"eval"
        metadata=json.loads((path/"metadata.json").read_text(encoding="utf-8"))
        if metadata["method"]!=job["method"] or metadata["seed"]!=job["seed"]:
            raise ValueError(f"Metadata identity mismatch: {path}")
        with (path/"evaluation.csv").open(encoding="utf-8") as stream:
            rows=list(csv.DictReader(stream))
        identities=[(r["scenario"],int(r["eval_seed"])) for r in rows]
        if len(rows)!=len(expected_rows) or set(identities)!=expected_rows or job["rows"]!=len(rows):
            raise ValueError(f"Missing or duplicate evaluation rows: {path}")
        for row in rows:
            if row["method"]!=job["method"] or int(row["train_seed"])!=job["seed"]:
                raise ValueError(f"CSV seed/method mismatch: {path}")
            for key,value in row.items():
                if key in ("method","scenario") or value in ("True","False"):
                    continue
                if not math.isfinite(float(value)):
                    raise ValueError(f"Nonfinite {key}: {path}")
        count+=len(rows)
    return dict(run=str(root.resolve()),jobs=len(jobs),evaluation_rows=count,
                complete=True,full_matrix_checked=require_full_matrix)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--require-full-matrix",action="store_true")
    args=parser.parse_args()
    print(json.dumps(audit_run(args.run,args.require_full_matrix),indent=2))


if __name__=="__main__": main()
