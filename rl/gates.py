from __future__ import annotations
import json
from project_config import ROOT,backend_source_hash


def require_edca_validation() -> dict:
    path=ROOT/"results/summary/edca_validation.json"
    if not path.exists(): raise RuntimeError("Stage 0 has not been run")
    result=json.loads(path.read_text(encoding="utf-8"))
    fingerprint=backend_source_hash()
    if not result["passed"] or result["source_hash"]!=fingerprint:
        raise RuntimeError("Stage 0 failed or backend changed; run experiments/run_edca_validation.py")
    return result
