#!/usr/bin/env python3
"""Create a Linux source archive without experiment outputs or build artifacts."""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TOP_LEVEL_FILES = (
    "project_config.py", "requirements.txt", "README_CN.md", "SERVER_LINUX.md",
)
SOURCE_DIRS = (
    "configs", "experiments", "models", "ns3_ext", "reporting", "rl", "scripts", "sim", "tests",
)
NS3_EXCLUDED_PARTS = {"build", "cmake-cache", "__pycache__", ".pytest_cache"}


def excluded(path: Path) -> bool:
    relative = path.relative_to(ROOT / "vendor" / "ns-3.44")
    return any(part in NS3_EXCLUDED_PARTS for part in relative.parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "exports" / "compress_DRL_EDCA_linux_source.tar.gz")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = "compress_DRL_EDCA"
    with tarfile.open(output, "w:gz", compresslevel=6) as archive:
        for name in TOP_LEVEL_FILES:
            archive.add(ROOT / name, arcname=f"{prefix}/{name}")
        for name in SOURCE_DIRS:
            archive.add(ROOT / name, arcname=f"{prefix}/{name}",
                        filter=lambda info: None if any(part in {"__pycache__", ".pytest_cache"}
                                                       for part in Path(info.name).parts) else info)
        ns3 = ROOT / "vendor" / "ns-3.44"
        if not ns3.is_dir():
            raise FileNotFoundError(ns3)
        for path in ns3.rglob("*"):
            if excluded(path):
                continue
            archive.add(path, arcname=f"{prefix}/vendor/ns-3.44/{path.relative_to(ns3).as_posix()}",
                        recursive=False)
    print(output)


if __name__ == "__main__":
    main()
