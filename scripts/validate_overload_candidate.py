"""Validate the proposed absolute-rate S4 candidate."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config
from rl.artifacts import save_json
from rl.gates import require_edca_validation
from rl.trainer import save_csv
from scripts.evaluate_mixed_load_candidates import run_case


OUTPUT = ROOT / "results/raw/overload_candidate_20261019"
RATES = (0.45, 1.00, 1.20, 0.25)
SEEDS = (501, 502, 503)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    gate = require_edca_validation(); config = load_config(ROOT / "configs/full.yaml")
    jobs = [(config, "overload_candidate", RATES, seed, str(OUTPUT)) for seed in SEEDS]
    save_json(OUTPUT / "metadata.json", dict(state="running", source_hash=gate["source_hash"],
              per_sta_mbps_by_ac=RATES, seeds=SEEDS, warmup_s=10, measure_s=20,
              queue_limits=config["queues"]["limits_packets"], queue_lifetime_s=10.0,
              retry_limit=7, started_at=datetime.now().astimezone().isoformat()))
    rows=[]; started=time.monotonic()
    with ProcessPoolExecutor(max_workers=config["run"]["workers"]) as pool:
        futures={pool.submit(run_case,job):job[3] for job in jobs}
        for future in as_completed(futures):
            row=future.result(); rows.append(row)
            save_csv(OUTPUT/"runs.csv",sorted(rows,key=lambda r:r["seed"]))
            print(json.dumps({"seed":futures[future],"drop_rate":row["drop_rate"],
                              "vo_p99_9_ms":row["vo_delay_p99_9_ms"],
                              "vi_p99_9_ms":row["vi_delay_p99_9_ms"]}),flush=True)
    meta=json.loads((OUTPUT/"metadata.json").read_text(encoding="utf-8"))
    meta.update(state="complete",runtime_s=time.monotonic()-started,
                completed_at=datetime.now().astimezone().isoformat())
    save_json(OUTPUT/"metadata.json",meta)


if __name__=="__main__": main()
