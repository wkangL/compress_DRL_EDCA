"""Enlarge only the active AC queue at calibrated load points."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
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
from scripts.calibrate_ac_load import run_case


OUTPUT = ROOT / "results/raw/ac_queue_sensitivity_20261019"
CASES = (("vo", 1.00), ("vo", 1.50), ("vi", 3.00))
SEEDS = (501, 502, 503)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    gate = require_edca_validation(); base = load_config(ROOT / "configs/full.yaml")
    jobs = []
    for ac, rate in CASES:
        config = deepcopy(base)
        index = 0 if ac == "vo" else 1
        config["queues"]["limits_packets"][index] *= 10
        for seed in SEEDS:
            jobs.append((config, ac, rate, seed, str(OUTPUT)))
    metadata = dict(state="running", source_hash=gate["source_hash"], cases=CASES,
                    seeds=SEEDS, multiplier=10, queue_lifetime_s=10.0,
                    retry_limit=7, started_at=datetime.now().astimezone().isoformat())
    save_json(OUTPUT / "metadata.json", metadata)
    rows = []; started = time.monotonic()
    with ProcessPoolExecutor(max_workers=base["run"]["workers"]) as pool:
        futures = {pool.submit(run_case, job): (job[1], job[2], job[3]) for job in jobs}
        for future in as_completed(futures):
            row = future.result(); rows.append(row)
            save_csv(OUTPUT / "runs.csv", sorted(rows, key=lambda r: (r["ac"], r["per_sta_target_mbps"], r["seed"])))
            print(json.dumps({"complete": futures[future], "drop_rate": row["drop_rate"],
                              "p99_9_ms": row["delay_p99_9_ms"],
                              "pending_end": row["pending_end"]}), flush=True)
    metadata.update(state="complete", runtime_s=time.monotonic() - started,
                    completed_at=datetime.now().astimezone().isoformat())
    save_json(OUTPUT / "metadata.json", metadata)


if __name__ == "__main__":
    main()
