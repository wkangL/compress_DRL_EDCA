"""Run Default-EDCA TGax load calibration with bounded concurrency."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import ctypes
from ctypes import wintypes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT


ALL_VARIANTS = ("candidate_ftp_bk", "candidate_voip8_gaming4_ftp_bk",
                "image_business_40sta",
                "image_vo_vi_single_bss",
                "candidate_full40_multiac_v1", "candidate_full40_multiac_v2",
                "candidate_full40_multiac_v3",
                "candidate_full40_multiac_v4",
                "candidate_full40_multiac_v5",
                "candidate_full40_multiac_v6",
                "mix_min1", "mix_min2", "load025",
                "load050", "load075", "load100", "load125", "load150", "rt125", "rt150")


def inputs_root(seed: int) -> Path:
    return (ROOT / "results/raw/tgax_s3_inputs_20261021" if seed == 20261021
            else ROOT / f"results/raw/tgax_s3_inputs_seed{seed}")


def ensure_inputs(seed: int, env: dict[str, str], scene: str) -> Path:
    base = inputs_root(seed)
    if not (base / f"{scene}/resolved_config.json").is_file():
        subprocess.run([sys.executable, str(ROOT / "scripts/build_tgax_s3_inputs.py"),
                        "--seed", str(seed), "--output", str(base)],
                       cwd=ROOT, env=env, check=True)
    if scene in {"subset7", "full19"}:
        subprocess.run([sys.executable, str(ROOT / "scripts/build_tgax_load_variants.py"),
                        "--seed", str(seed), "--inputs-root", str(base), "--scene", scene],
                       cwd=ROOT, env=env, check=True)
    return base


def run_one(exe: Path, env: dict[str, str], output: Path, base: Path,
            variant: str, seed: int, warmup: float, duration: float, drain: float,
            backend: str, profile: str, scene: str, expected_stas: int,
            trace_level: str) -> dict:
    if output.exists() and (output / "calibration_validation.json").is_file():
        status = json.loads((output / "calibration_validation.json").read_text(encoding="utf-8"))
        if status.get("status") == "PASS":
            saved = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            requested = {"scene": scene, "variant": variant, "seed": seed, "profile": profile,
                         "warmup_s": warmup, "measurement_s": duration, "drain_s": drain,
                         "phy_backend": backend, "trace_level": trace_level}
            mismatch = {key: (saved.get(key), value) for key, value in requested.items()
                        if saved.get(key) != value}
            if mismatch:
                raise RuntimeError(f"Existing PASS run has different settings: {mismatch}")
            return {"variant": variant, "seed": seed, "profile": profile,
                    "status": "SKIP_PASS", "seconds": 0.0, "output": str(output)}
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite incomplete output: {output}")
    output.mkdir(parents=True)
    flows = base / f"{scene}/{variant}/flows.csv"
    command = [str(exe), f"--topology={base / scene / 'topology.csv'}", f"--flows={flows}",
               f"--output={output}", f"--warmup={warmup}", f"--duration={duration}",
               f"--drain={drain}", f"--seed={seed}", f"--traceLevel={trace_level}",
               f"--phyBackend={backend}", f"--targetProfile={profile}"]
    started = time.perf_counter()
    peak_rss = 0
    with (output / "stdout.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
        handle = kernel32.OpenProcess(0x0400 | 0x0010, False, process.pid)
        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        while process.poll() is None:
            if handle and psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                peak_rss = max(peak_rss, int(counters.PeakWorkingSetSize))
            time.sleep(0.25)
        if handle:
            kernel32.CloseHandle(handle)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    elapsed = time.perf_counter() - started
    manifest = {
        "scene": scene, "variant": variant, "seed": seed, "profile": profile,
        "warmup_s": warmup, "measurement_s": duration, "drain_s": drain,
        "phy_backend": backend, "trace_level": trace_level, "rl_enabled": False,
        "elapsed_wall_seconds": elapsed, "peak_rss_bytes": peak_rss,
        "peak_rss_mib": peak_rss / 1024 / 1024,
        "started_by": "run_tgax_calibration_matrix.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copy2(base / f"{scene}/{variant}/manifest.json", output / "load_manifest.json")
    subprocess.run([sys.executable, str(ROOT / "scripts/audit_tgax_calibration_run.py"),
                    "--run", str(output), "--flows", str(flows),
                    "--expected-stas", str(expected_stas), "--scene", scene],
                   cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    return {"variant": variant, "seed": seed, "profile": profile,
            "status": "PASS", "seconds": elapsed, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default=",".join(ALL_VARIANTS))
    parser.add_argument("--seeds", default="20261021")
    parser.add_argument("--profiles", default="default")
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--drain", type=float, default=0.5)
    parser.add_argument("--backend", choices=("yans", "spectrum"), default="yans")
    parser.add_argument("--scene", choices=("api1", "subset7", "full19"), default="subset7")
    parser.add_argument("--trace-level", choices=("summary", "full"), default="summary")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--output-root", type=Path,
                        default=None)
    args = parser.parse_args()
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    profiles = tuple(item.strip() for item in args.profiles.split(",") if item.strip())
    unknown = set(variants) - set(ALL_VARIANTS)
    if unknown:
        raise SystemExit(f"Unknown variants: {sorted(unknown)}")
    env = dict(os.environ)
    ns3 = ROOT / "vendor/ns-3.44"
    env["PATH"] = os.pathsep.join([str(ROOT / ".tools/mingw64/bin"), str(ns3 / "build/lib"), env.get("PATH", "")])
    expected_stas = {"api1": 40, "subset7": 280, "full19": 760}[args.scene]
    output_root = args.output_root or ROOT / f"results/raw/tgax_{args.scene}_calibration_20261022"
    bases = {seed: ensure_inputs(seed, env, args.scene) for seed in seeds}
    exe = ns3 / "build/scratch/ns3.44-tgax-s3.exe"
    jobs = []
    for variant in variants:
        for seed in seeds:
            for profile in profiles:
                output = output_root / variant / f"seed{seed}" / profile
                jobs.append((output, bases[seed], variant, seed, profile))
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, exe, env, output, base, variant, seed,
                               args.warmup, args.duration, args.drain, args.backend, profile,
                               args.scene, expected_stas, args.trace_level)
                   for output, base, variant, seed, profile in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    (output_root / "matrix_status.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
