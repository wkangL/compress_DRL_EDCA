#!/usr/bin/env python3
"""Recover final-only evaluation after all three 600-step trainings finish."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT

RUN = ROOT / "results/raw/mac_arrival40_voip40_20_gaming80_40_bv3ul10_cmdp_600step_3seed_20261030"
CONFIG = ROOT / "configs/mac_arrival40_voip40_20_gaming80_40_bv3ul10_cmdp_calibrated.generated.json"
SEEDS = (42, 66, 88)


def atomic_status(value: dict) -> None:
    temporary = RUN / "status.recovery.tmp"
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(RUN / "status.json")


def main() -> None:
    deadline = time.monotonic() + 6 * 3600
    snapshots = [RUN / f"seed_{seed}/snapshots/step_600.pt" for seed in SEEDS]
    while not all(path.is_file() for path in snapshots):
        if time.monotonic() > deadline:
            raise TimeoutError("Timed out waiting for final 600-step snapshots")
        time.sleep(60)
    atomic_status({"state": "final_evaluation_recovery", "completed_training_seeds": list(SEEDS)})
    processes = []
    streams = []
    for seed in SEEDS:
        root = RUN / f"seed_{seed}"
        status_file = root / "status.json"
        if status_file.is_file() and json.loads(status_file.read_text(encoding="utf-8")).get("state") == "complete":
            continue
        stream = (RUN / f"seed_{seed}_finalization.log").open("a", encoding="utf-8")
        process = subprocess.Popen([
            sys.executable, str(ROOT / "scripts/finalize_tgax_ddqn_final_only.py"),
            "--config", str(CONFIG), "--root", str(root),
        ], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        processes.append((seed, process)); streams.append(stream)
    failures = []
    for seed, process in processes:
        returncode = process.wait()
        if returncode:
            failures.append({"seed": seed, "returncode": returncode})
    for stream in streams:
        stream.close()
    if failures:
        atomic_status({"state": "final_evaluation_aborted", "failures": failures})
        raise RuntimeError(failures)
    with (RUN / "analysis_recovery.log").open("a", encoding="utf-8") as stream:
        completed = subprocess.run([
            sys.executable,
            str(ROOT / "scripts/analyze_mac_arrival40_voip40_20_gaming80_40_bv3ul10_training.py"),
            "--root", str(RUN),
        ], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        atomic_status({"state": "analysis_aborted", "returncode": completed.returncode})
        raise RuntimeError("Analysis failed")
    atomic_status({
        "state": "complete", "train_seeds": list(SEEDS), "training_steps_per_seed": 600,
        "final_checkpoint_only": True,
        "report": "reports/单BSS新业务配置-CMDP-TCN-DDQN三seed训练报告.md",
        "recovered_post_training_finalization": True,
    })


if __name__ == "__main__":
    main()
