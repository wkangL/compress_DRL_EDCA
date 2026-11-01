#!/usr/bin/env python3
"""Refresh a live training PNG for a process started before live plotting existed."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_tgax_ddqn_extended import save_live_diagnostics


def rows(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--maximum-hours", type=float, default=24.0)
    args = parser.parse_args()
    root = args.root.resolve()
    deadline = time.monotonic() + args.maximum_hours * 3600
    last_signature = None
    while time.monotonic() < deadline:
        step_path, loss_path = root / "training_steps.csv", root / "losses.csv"
        signature = tuple((path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else (0, 0)
                          for path in (step_path, loss_path))
        if signature != last_signature and step_path.is_file():
            save_live_diagnostics(root, rows(step_path), rows(loss_path))
            last_signature = signature
        status_path = root / "status.json"
        if status_path.is_file():
            state = json.loads(status_path.read_text(encoding="utf-8")).get("state")
            if state in ("complete", "failed"):
                return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
