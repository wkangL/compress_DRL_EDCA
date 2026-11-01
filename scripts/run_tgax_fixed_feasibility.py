#!/usr/bin/env python3
"""Paired fixed-action TGax feasibility runs with resumable per-run outputs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config
from scripts.run_tgax_fixed_actions import flatten, run_one, save_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-file", type=Path,
                        default=ROOT / "configs/tgax_fixed_feasibility_candidates.json")
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    config = deepcopy(load_config(ROOT / "configs/tgax_ddqn_reward_v3.yaml"))
    definition = json.loads(args.candidate_file.resolve().read_text(encoding="utf-8"))
    candidates = definition["candidates"]
    if args.only:
        missing = set(args.only) - set(candidates)
        if missing: raise ValueError(f"Unknown candidates: {sorted(missing)}")
        candidates = {key: candidates[key] for key in args.only}
    root = ROOT / "results/raw" / args.tag
    root.mkdir(parents=True, exist_ok=True)
    experiment = {"state": "running", "duration_s": args.duration, "seeds": args.seeds,
                  "workers": args.workers, "environment": config["identity"]["environment"],
                  "candidate_file": str(args.candidate_file.resolve()), "candidates": candidates}
    (root / "experiment_config.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")

    def execute(seed: int, name: str, spec: dict) -> dict:
        seed_root = root / f"seed_{seed}"; seed_root.mkdir(exist_ok=True)
        done = seed_root / name / "fixed_action_summary.json"
        if done.is_file():
            return json.loads(done.read_text(encoding="utf-8"))
        if (seed_root / name).exists():
            raise RuntimeError(f"Incomplete run directory requires inspection: {seed_root / name}")
        return run_one(config, name, spec, seed, args.duration, seed_root)

    jobs = [(seed, name, spec) for seed in args.seeds for name, spec in candidates.items()]
    results = []; started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, *job): job for job in jobs}
        for future in as_completed(futures):
            seed, name, _ = futures[future]
            result = future.result(); results.append(result)
            status = {"state": "running", "completed_runs": len(results), "total_runs": len(jobs),
                      "last_seed": seed, "last_action": name, "runtime_s": time.perf_counter()-started}
            (root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            save_csv(root / "comparison.csv", [flatten(r) for r in sorted(results, key=lambda x:(x["seed"],x["name"]))])
            print(json.dumps(status), flush=True)

    audits = {}
    for seed in args.seeds:
        group = [r for r in results if r["seed"] == seed]
        demand = [r["application_generated_by_step"] for r in group]
        audits[str(seed)] = {
            "paired_generated_demand_equal_each_step": all(x == demand[0] for x in demand[1:]),
            "all_associations_correct": all(r["audit"]["association_mismatches"] == 0 for r in group),
            "all_neighbors_default": all(r["audit"]["neighbor_nondefault_rows"] == 0 for r in group),
            # Time 0 is the common Default traffic warmup and time 1 is the
            # first measured action application.  No later STEP may rewrite it.
            "constant_action_not_rewritten": all(all(
                int(changed) == 0 for time_s, changed in r["audit"]["action_changed_by_step"].items()
                if float(time_s) > 1.0 + 1e-9) for r in group),
        }
    if not all(all(check.values()) for check in audits.values()):
        raise RuntimeError(f"Paired audit failed: {audits}")
    final = {"state": "complete", "completed_runs": len(results), "total_runs": len(jobs),
             "runtime_s": time.perf_counter()-started, "audits": audits}
    (root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (root / "paired_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    (root / "status.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
