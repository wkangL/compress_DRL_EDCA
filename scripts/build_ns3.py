"""Patch and build the pinned ns-3 source on Windows or Linux."""
from __future__ import annotations
from pathlib import Path
import os, shutil, subprocess, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT
from ns3_ext.patch_ns3 import patch
from sim.backend.ns3_backend import runtime_env


def main() -> None:
    root=ROOT/"vendor/ns-3.44"
    if not (root/"CMakeLists.txt").exists():
        raise SystemExit("Missing ns-3.44. See README_CN.md for the pinned clone command.")
    commit=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if commit!="43dce6710b8df69685e3479c1d33a571dda714ea":
        raise SystemExit("Unexpected ns-3 source commit; refusing unreviewed patch")
    patch(root)
    for name in ("wifi-control.cc", "tgax-control.cc", "tgax-s3.cc"):
        shutil.copy2(ROOT / "ns3_ext" / name, root / "scratch" / name)
    env=runtime_env()
    cmake=shutil.which("cmake",path=env["PATH"])
    if cmake is None: raise SystemExit("CMake/compiler unavailable")
    generator = "Ninja" if shutil.which("ninja", path=env["PATH"]) else "Unix Makefiles"
    subprocess.run([cmake,"-S",str(root),"-B",str(root/"cmake-cache"),"-G",generator,
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        "-DNS3_ENABLED_MODULES=applications;core;flow-monitor;internet;mobility;network;propagation;spectrum;wifi",
        "-DNS3_TESTS=ON","-DNS3_EXAMPLES=OFF","-DNS3_PYTHON_BINDINGS=OFF","-DNS3_WARNINGS_AS_ERRORS=OFF",
        "-DNS3_CCACHE=OFF",f"-DPYTHON3={sys.executable}"],env=env,check=True)
    jobs = str(min(16, os.cpu_count() or 8))
    subprocess.run([cmake,"--build",str(root/"cmake-cache"),"--parallel",jobs,"--target",
                    "scratch_tgax-control","scratch_wifi-control","test-runner"],env=env,check=True)
    # CMake's Windows profile suffix is not present in the configured backend
    # path.  Refresh the stable executable names after every successful build
    # so experiments cannot silently run an older binary.
    if os.name == "nt":
        for name in ("tgax-control", "wifi-control"):
            built=root/"build"/"scratch"/f"ns3.44-{name}-relwithdebinfo.exe"
            stable=root/"build"/"scratch"/f"ns3.44-{name}.exe"
            if not built.is_file():
                raise FileNotFoundError(built)
            shutil.copy2(built,stable)


if __name__=="__main__": main()
