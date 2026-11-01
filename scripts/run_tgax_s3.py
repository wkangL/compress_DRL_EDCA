"""Build and run the ns-3.44 TGax Scenario 3 environment on Windows."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT


DEFAULTS = {
    ("main3", "smoke"): (1.0, 1.0, 0.5),
    ("main3", "full"): (1.0, 5.0, 1.0),
    ("api1", "smoke"): (1.0, 1.0, 0.5),
    ("subset7", "smoke"): (1.2, 0.2, 0.2),
    ("subset7", "full"): (1.5, 5.0, 1.0),
    ("full19", "smoke"): (2.0, 0.2, 0.2),
    ("full19", "full"): (2.0, 5.0, 1.0),
}


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("main3", "api1", "subset7", "full19"), default="main3")
    p.add_argument("--traffic", choices=("smoke", "full"), default="smoke")
    p.add_argument("--output", type=Path)
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument("--warmup", type=float)
    p.add_argument("--duration", type=float)
    p.add_argument("--drain", type=float)
    p.add_argument("--skip-build", action="store_true")
    args = p.parse_args()
    if args.variant == "api1" and args.traffic == "full":
        raise SystemExit("api1 is a protocol smoke variant; use --traffic smoke")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (args.output or ROOT / f"results/raw/tgax_s3_{args.variant}_{args.traffic}_{stamp}").resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    warmup, duration, drain = DEFAULTS[(args.variant, args.traffic)]
    warmup = args.warmup if args.warmup is not None else warmup
    duration = args.duration if args.duration is not None else duration
    drain = args.drain if args.drain is not None else drain

    ns3 = ROOT / "vendor/ns-3.44"
    py313 = Path(r"C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe")
    launcher = py313 if py313.is_file() else Path(sys.executable)
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(ROOT / ".tools/mingw64/bin"),
                                   str(ns3 / "build/lib"), env.get("PATH", "")])
    inputs_base = (ROOT / "results/raw/tgax_s3_inputs_20261021" if args.seed == 20261021
                   else ROOT / f"results/raw/tgax_s3_inputs_seed{args.seed}")
    if not (inputs_base / args.variant / "resolved_config.json").is_file():
        run([sys.executable, str(ROOT / "scripts/build_tgax_s3_inputs.py"),
             "--seed", str(args.seed), "--output", str(inputs_base)], ROOT, env)
    if not args.skip_build:
        shutil.copy2(ROOT / "ns3_ext/tgax-s3.cc", ns3 / "scratch/tgax-s3.cc")
        run([str(launcher), "ns3", "configure", "-d", "release", "--disable-tests",
             "--enable-modules=wifi,mobility,propagation,network,core,internet,applications,flow-monitor,spectrum"], ns3, env)
        run([str(launcher), "ns3", "build", "tgax-s3"], ns3, env)
    exe = ns3 / "build/scratch/ns3.44-tgax-s3.exe"
    inputs = inputs_base / args.variant
    flows = inputs / ("protocol_smoke_flows.csv" if args.traffic == "smoke" else "flows.csv")
    run([str(exe), f"--topology={inputs / 'topology.csv'}", f"--flows={flows}",
         f"--output={output}", f"--warmup={warmup}", f"--duration={duration}",
         f"--drain={drain}", f"--seed={args.seed}"], ROOT, env)
    shutil.copy2(inputs / "resolved_config.json", output / "input_resolved_config.json")
    shutil.copy2(inputs / "topology.csv", output / "input_topology.csv")
    shutil.copy2(flows, output / "input_flows.csv")
    (output / "run_manifest.json").write_text(json.dumps({
        "variant": args.variant, "traffic": args.traffic, "seed": args.seed,
        "warmup_s": warmup, "measurement_s": duration, "drain_s": drain,
        "ns3_version": "3.44", "build_profile": "release",
        "rl_enabled": False, "source": str((ROOT / "ns3_ext/tgax-s3.cc").resolve()),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    run([sys.executable, str(ROOT / "scripts/validate_tgax_s3_run.py"),
         "--run", str(output), "--topology", str(inputs / "topology.csv"),
         "--flows", str(flows)], ROOT, env)
    print(f"Completed: {output}")


if __name__ == "__main__":
    main()
