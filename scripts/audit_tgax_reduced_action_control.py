#!/usr/bin/env python3
"""Two-step smoke test for the 865-action TGax controller and native fallback."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config
from rl.tgax_environment import TgaxControlEnvironment


NATIVE = {
    "VO": (2, 3, 7, 2080), "VI": (2, 7, 15, 4096),
    "BE": (3, 15, 1023, 0), "BK": (7, 15, 1023, 0),
}


def params(row: dict[str, str]) -> tuple[int, int, int, int]:
    return tuple(int(float(row[key])) for key in ("aifsn", "cwmin", "cwmax", "txop_us"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    config = deepcopy(load_config(ROOT / "configs/tgax_ddqn_full40_multiac_v6.yaml"))
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    fallback = np.asarray(config["action"]["default_action"], dtype=np.int64)
    custom = np.asarray([8, 1, 0, 2, 1], dtype=np.int64)
    env = TgaxControlEnvironment(config, "compressed_async_age", args.seed, args.output, 3.0)
    try:
        env.reset_observation()
        _, fallback_reward, _ = env.step(fallback, 1.0, custom=True)
        _, custom_reward, _ = env.step(custom, 1.0, custom=True)
        _, restored_reward, _ = env.step(fallback, 1.0, custom=True)
    finally:
        env.close()

    rows = list(csv.DictReader((args.output / "edca_step_runtime.csv").open(encoding="utf-8")))
    target = [row for row in rows if row["bss"] == "0"]
    neighbors = [row for row in rows if row["bss"] != "0"]
    fallback_rows = [row for row in target if abs(float(row["time_s"]) - 1.0) < 1e-6]
    custom_rows = [row for row in target if abs(float(row["time_s"]) - 2.0) < 1e-6]
    restored_rows = [row for row in target if abs(float(row["time_s"]) - 3.0) < 1e-6]
    checks = {
        "fallback_logged_native_growth": bool(fallback_rows) and all(row["custom_growth"] == "0" for row in fallback_rows),
        "fallback_logged_p0": bool(fallback_rows) and all(row["profile"] == "0" for row in fallback_rows),
        "fallback_edca_is_default": bool(fallback_rows) and all(params(row) == NATIVE[row["ac"]] for row in fallback_rows),
        "custom_logged_growth": bool(custom_rows) and all(row["custom_growth"] == "1" for row in custom_rows),
        "custom_logged_p8": bool(custom_rows) and all(row["profile"] == "8" for row in custom_rows),
        "fallback_restores_native_growth": bool(restored_rows) and all(row["custom_growth"] == "0" for row in restored_rows),
        "fallback_restores_p0": bool(restored_rows) and all(row["profile"] == "0" for row in restored_rows),
        "fallback_restores_default_edca": bool(restored_rows) and all(params(row) == NATIVE[row["ac"]] for row in restored_rows),
        "neighbors_remain_default": bool(neighbors) and all(
            row["custom_growth"] == "0" and params(row) == NATIVE[row["ac"]] for row in neighbors),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "requested_fallback": fallback.tolist(),
        "requested_custom": custom.tolist(),
        "fallback_reward": fallback_reward,
        "custom_reward": custom_reward,
        "restored_reward": restored_reward,
        "target_runtime_rows": len(target),
        "neighbor_runtime_rows": len(neighbors),
    }
    (args.output / "reduced_action_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
