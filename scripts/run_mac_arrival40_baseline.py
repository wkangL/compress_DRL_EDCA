"""Build and run the one-BSS direct-MAC TGax-arrival Default baseline."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config
from sim.backend.ns3_backend import runtime_env


CONFIG = ROOT / "configs/mac_arrival40_default.yaml"
INPUT = ROOT / "results/raw/tgax_mac_arrival40_inputs_20261029"
SMOKE = ROOT / "results/raw/mac_arrival40_default_smoke_seed41_r2_20261029"
FORMAL = ROOT / "results/raw/mac_arrival40_default_20s_3seed_20261029"


def invoke(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def build(env: dict[str, str]) -> Path:
    ns3 = ROOT / "vendor/ns-3.44"
    shutil.copy2(ROOT / "ns3_ext/tgax-s3.cc", ns3 / "scratch/tgax-s3.cc")
    py313 = Path(r"C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe")
    invoke([str(py313 if py313.is_file() else Path(sys.executable)), "ns3", "build", "tgax-s3"], ns3, env)
    executable = ns3 / "build/scratch/ns3.44-tgax-s3.exe"
    if not executable.is_file():
        raise FileNotFoundError(executable)
    return executable


def run_one(executable: Path, output: Path, seed: int, duration: float,
            trace_level: str, config: dict, env: dict[str, str]) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    run_cfg = config["run"]
    traffic = config["traffic"]
    command = [
        str(executable), f"--topology={INPUT / 'topology.csv'}",
        f"--flows={INPUT / 'flows.csv'}", f"--output={output}",
        f"--seed={seed}", f"--warmup={run_cfg['association_warmup_s']}",
        f"--saturationWarmup={run_cfg['source_warmup_s']}", f"--duration={duration}",
        f"--drain={run_cfg['drain_s']}", f"--traceLevel={trace_level}",
        "--phyBackend=spectrum", "--targetProfile=default", "--trafficMode=mac-arrival",
        "--saturationLowWatermark=32",
        f"--saturationHighWatermark={traffic['queue_admission_high_watermark_packets']}",
        f"--saturationRefillUs={traffic['admission_poll_us']}",
    ]
    started = time.perf_counter()
    invoke(command, ROOT, env)
    manifest = {
        "seed": seed, "duration_s": duration,
        "wall_clock_s": time.perf_counter() - started,
        "config": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "topology": str((INPUT / "topology.csv").relative_to(ROOT)).replace("\\", "/"),
        "flows": str((INPUT / "flows.csv").relative_to(ROOT)).replace("\\", "/"),
        "traffic_mode": "mac-arrival", "internet_stack": False,
        "backend_source": "ns3_ext/tgax-s3.cc", "ddqn": False, "reward": None,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "formal", "all"), default="all")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    config = load_config(CONFIG)
    env = runtime_env()
    env["PATH"] = os.pathsep.join([str(ROOT / ".tools/mingw64/bin"),
                                   str(ROOT / "vendor/ns-3.44/build/lib"), env.get("PATH", "")])
    invoke([sys.executable, str(ROOT / "scripts/build_mac_arrival40_input.py")], ROOT, env)
    executable = (ROOT / "vendor/ns-3.44/build/scratch/ns3.44-tgax-s3.exe"
                  if args.skip_build else build(env))

    if args.phase in {"smoke", "all"}:
        run_one(executable, SMOKE, 41, 5.0, "full", config, env)
    if args.phase in {"formal", "all"}:
        FORMAL.mkdir(parents=True, exist_ok=True)
        seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
        def task(seed: int) -> dict:
            return run_one(executable, FORMAL / f"seed_{seed}", seed,
                           float(config["run"]["measurement_s"]),
                           str(config["run"]["trace_level"]), config, env)
        with ThreadPoolExecutor(max_workers=min(args.workers, len(seeds))) as pool:
            completed = list(pool.map(task, seeds))
        (FORMAL / "status.json").write_text(json.dumps({
            "state": "complete", "seeds": seeds, "runs": completed,
        }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
