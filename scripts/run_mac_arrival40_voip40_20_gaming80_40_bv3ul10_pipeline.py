#!/usr/bin/env python3
"""Calibrate, freeze and run the requested three-seed 600-step CMDP study.

The pipeline intentionally performs no checkpoint selection.  Each learned
policy is evaluated only after training, using its final checkpoint paired with
Default on the same external-demand seeds.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT


BASE_CONFIG = ROOT / "configs/mac_arrival40_voip40_20_gaming80_40_bv3ul10_cmdp.yaml"
CALIBRATION = ROOT / "results/raw/mac_arrival40_voip40_20_gaming80_40_bv3ul10_default_calibration_20261030"
CALIBRATED_CONFIG = ROOT / "configs/mac_arrival40_voip40_20_gaming80_40_bv3ul10_cmdp_calibrated.generated.json"
PIPELINE_ROOT = ROOT / "results/raw/mac_arrival40_voip40_20_gaming80_40_bv3ul10_cmdp_600step_3seed_20261030"
TRAIN_SEEDS = (42, 66, 88)
CALIBRATION_SEEDS = (41, 43, 45)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def command(args: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        completed = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(args)}")


def freeze_calibration() -> None:
    data = json.loads((CALIBRATION / "calibration.json").read_text(encoding="utf-8"))
    if data["status"] != "PASS":
        raise RuntimeError("Default calibration did not pass")
    frozen = {
        "extends": BASE_CONFIG.name,
        "identity": {"purpose": "calibrated three-seed 600-step final-checkpoint-only CMDP training"},
        "reward": {
            "low_priority_min_satisfaction": data["recommended_low_priority_min_satisfaction"],
            "budgets": data["recommended_budgets"],
            "budget_status": "frozen from independent Default one-second STEP calibration",
            "budget_source": str((CALIBRATION / "calibration.json").relative_to(ROOT)).replace("\\", "/"),
        },
    }
    atomic_json(CALIBRATED_CONFIG, frozen)


def run_training(seed: int) -> tuple[subprocess.Popen, object, Path]:
    root = PIPELINE_ROOT / f"seed_{seed}"
    status_path = root / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") == "complete":
            return None, None, root
    final_snapshot = root / "snapshots/step_600.pt"
    if final_snapshot.is_file():
        args = [sys.executable, str(ROOT / "scripts/finalize_tgax_ddqn_final_only.py"),
                "--config", str(CALIBRATED_CONFIG), "--root", str(root)]
    else:
        args = [sys.executable, str(ROOT / "scripts/run_tgax_ddqn_extended.py"),
                "--config", str(CALIBRATED_CONFIG), "--root", str(root), "--seed", str(seed),
                "--scaler-events", str(CALIBRATION / "seed_41/events.log")]
        if (root / "train/checkpoint.pt").is_file():
            args.append("--resume")
    # Keep the launcher log outside the experiment root: the trainer correctly
    # refuses to overwrite any pre-existing non-resume directory.
    log = PIPELINE_ROOT / f"seed_{seed}_pipeline_train.log"
    stream = log.open("a", encoding="utf-8")
    return subprocess.Popen(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT), stream, root


def main() -> None:
    PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(PIPELINE_ROOT / "status.json", {
        "state": "default_calibration", "train_seeds": TRAIN_SEEDS,
        "calibration_seeds": CALIBRATION_SEEDS, "training_steps_per_seed": 600,
        "evaluation": "final checkpoint versus paired Default only",
    })
    calibration_status = CALIBRATION / "status.json"
    calibration_complete = (calibration_status.is_file() and
        json.loads(calibration_status.read_text(encoding="utf-8")).get("state") == "complete")
    if not calibration_complete:
        command([sys.executable, str(ROOT / "scripts/calibrate_mac_arrival40_cmdp.py"),
                 "--config", str(BASE_CONFIG), "--output", str(CALIBRATION),
                 "--duration", "20", "--workers", "3", "--seeds",
                 *(str(seed) for seed in CALIBRATION_SEEDS)],
                PIPELINE_ROOT / "calibration.log")
    freeze_calibration()

    atomic_json(PIPELINE_ROOT / "status.json", {
        "state": "training", "train_seeds": TRAIN_SEEDS,
        "training_steps_per_seed": 600, "completed": [],
    })
    running = [run_training(seed) for seed in TRAIN_SEEDS]
    while any(process is not None and process.poll() is None for process, _, _ in running):
        completed = []
        progress = {}
        for seed, (process, _, root) in zip(TRAIN_SEEDS, running):
            status_file = root / "status.json"
            if status_file.is_file():
                status = json.loads(status_file.read_text(encoding="utf-8"))
                progress[str(seed)] = status
                if status.get("state") == "complete":
                    completed.append(seed)
        atomic_json(PIPELINE_ROOT / "status.json", {
            "state": "training", "train_seeds": TRAIN_SEEDS,
            "training_steps_per_seed": 600, "completed": completed,
            "progress": progress,
        })
        time.sleep(60)
    errors = []
    for seed, (process, stream, root) in zip(TRAIN_SEEDS, running):
        if stream is not None:
            stream.close()
        if process is not None and process.returncode:
            errors.append({"seed": seed, "returncode": process.returncode,
                           "log": str((PIPELINE_ROOT / f"seed_{seed}_pipeline_train.log").relative_to(ROOT))})
    if errors:
        atomic_json(PIPELINE_ROOT / "status.json", {"state": "aborted", "errors": errors})
        raise RuntimeError(errors)

    command([sys.executable,
             str(ROOT / "scripts/analyze_mac_arrival40_voip40_20_gaming80_40_bv3ul10_training.py"),
             "--root", str(PIPELINE_ROOT)], PIPELINE_ROOT / "analysis.log")
    atomic_json(PIPELINE_ROOT / "status.json", {
        "state": "complete", "train_seeds": TRAIN_SEEDS,
        "training_steps_per_seed": 600,
        "report": "reports/单BSS新业务配置-CMDP-TCN-DDQN三seed训练报告.md",
    })


if __name__ == "__main__":
    main()
