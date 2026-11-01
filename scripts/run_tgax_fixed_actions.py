"""Run paired fixed-action TGax experiments without invoking DDQN."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config
from rl.tgax_environment import TgaxControlEnvironment


ACTIONS = {
    "A_default_beb": {"action": [0, 0, 5, 5, 2], "custom": False,
                      "description": "P0 + native BEB"},
    "B_default_custom": {"action": [0, 1, 3, 9, 1], "custom": True,
                         "description": "P0 + alpha2 beta1=1.6 beta2=2.8 gamma6"},
    "C_p8_beb": {"action": [8, 0, 5, 5, 2], "custom": False,
                  "description": "P8 + native BEB"},
    "D_p8_custom": {"action": [8, 1, 3, 9, 1], "custom": True,
                     "description": "P8 + alpha2 beta1=1.6 beta2=2.8 gamma6"},
}
ACS = ("vo", "vi", "be", "bk")
DIRECTIONS = ("ul", "dl")


def save_csv(path: Path, data: list[dict]) -> None:
    if not data:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader(); writer.writerows(data)


def jain_from_flows(run: Path, flows_path: Path) -> float | None:
    assignments = {int(r["flow_id"]): int(r["sta"])
                   for r in csv.DictReader(flows_path.open(encoding="utf-8"))
                   if int(r["bss"]) == 0}
    offered, received = Counter(), Counter()
    for row in csv.DictReader((run / "flow_summary.csv").open(encoding="utf-8")):
        fid = int(row["flow_id"])
        if int(row["bss"]) != 0 or fid not in assignments:
            continue
        sta = assignments[fid]
        offered[sta] += int(row["offered_bytes"])
        received[sta] += int(row["received_bytes"])
    values = np.asarray([min(received[s] / offered[s], 1.0) for s in offered if offered[s] > 0])
    denominator = len(values) * np.square(values).sum()
    return float(values.sum() ** 2 / denominator) if denominator > 0 else None


def application_delivery(run: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for row in csv.DictReader((run / "flow_summary.csv").open(encoding="utf-8")):
        if int(row["bss"]) != 0:
            continue
        key = f"{row['service']}_{row['direction']}"
        item = result.setdefault(key, {"offered_bytes": 0, "received_bytes": 0})
        item["offered_bytes"] += int(row["offered_bytes"])
        item["received_bytes"] += int(row["received_bytes"])
    return result


def summarize(metric_rows: list[dict], duration: float, run: Path, config: dict) -> dict:
    online_jain=float(np.mean([m["jain_demand_satisfaction"] for m in metric_rows]))
    direct=config.get("backend",{}).get("traffic_mode")=="mac-arrival"
    out: dict[str, object] = {"reward_mean": float(np.mean([m["reward"] for m in metric_rows])),
                             "jain_demand_satisfaction": (online_jain if direct else
                                 jain_from_flows(run, ROOT / config["backend"]["flows"])),
                             "application_generated_bytes": np.sum([m["application_generated_bytes"] for m in metric_rows], axis=0).tolist(),
                             "application_submitted_bytes": np.sum([m["application_submitted_bytes"] for m in metric_rows], axis=0).tolist(),
                             "application_received_bytes": np.sum([m["application_received_bytes"] for m in metric_rows], axis=0).tolist(),
                             "application_demand_debt_bytes": metric_rows[-1]["application_demand_debt_bytes"],
                             "application_generated_by_step": [m["application_generated_bytes"] for m in metric_rows]}
    received=np.asarray(out["application_received_bytes"],dtype=float)
    generated=np.asarray(out["application_generated_bytes"],dtype=float)
    available=(np.asarray(out["application_demand_debt_bytes"],dtype=float)+received
               if direct else generated)
    for index,ac in enumerate(ACS):
        out[f"{ac}_demand_satisfaction"]=float(min(received[index]/available[index],1.0)) if available[index]>0 else None
    reward_keys = sorted({key for row in metric_rows for key in row
                          if key.startswith("reward_") and key != "reward"})
    for key in reward_keys:
        out[f"{key}_mean"] = float(np.mean([m[key] for m in metric_rows if key in m]))
    cmdp_keys=sorted({key for row in metric_rows for key in row
                      if key.startswith("cmdp_") and key not in {"cmdp_costs","cmdp_multipliers"}
                      and isinstance(row.get(key),(int,float,bool))})
    for key in cmdp_keys:
        out[f"{key}_mean"]=float(np.mean([m[key] for m in metric_rows if key in m]))
    out["online_jain_demand_satisfaction_mean"] = float(np.mean(
        [m["jain_demand_satisfaction"] for m in metric_rows]))
    out["online_jain_active_sta_count_mean"] = float(np.mean(
        [m["jain_active_sta_count"] for m in metric_rows]))
    for ac in ACS:
        eligible = sum(int(m[f"{ac}_deadline_eligible"]) for m in metric_rows)
        delivered = sum(int(m[f"{ac}_on_time_delivered"]) for m in metric_rows)
        out[f"{ac}_on_time_delivery_rate"] = delivered / eligible if eligible else None
        out[f"{ac}_deadline_eligible"] = eligible
        for direction in DIRECTIONS:
            p = f"{direction}_{ac}"
            samples = sum((m[f"{p}_delay_samples_ms"] for m in metric_rows), [])
            risk = int(metric_rows[0][f"{p}_pending_start"]) + sum(int(m[f"{p}_arrived"]) for m in metric_rows)
            attempts = sum(int(m[f"{p}_tx_attempts"]) for m in metric_rows)
            failures = sum(int(m[f"{p}_failed"]) for m in metric_rows)
            retries = sum(int(m[f"{p}_retransmissions"]) for m in metric_rows)
            out.update({
                f"{p}_delay_p95_ms": float(np.percentile(samples, 95)) if samples else None,
                f"{p}_successful_samples": len(samples),
                f"{p}_throughput_mbps": float(np.mean([m[f"{p}_goodput_mbps"] for m in metric_rows])),
                f"{p}_final_drop_rate": failures / risk if risk else None,
                f"{p}_retransmission_rate": retries / attempts if attempts else None,
                f"{p}_risk_packets": risk,
                f"{p}_pending_end": int(metric_rows[-1][f"{p}_pending_end"]),
            })
    out["application_delivery"] = application_delivery(run)
    return out


def audit(run: Path, action: list[int], custom: bool) -> dict:
    associations = list(csv.DictReader((run / "associations.csv").open(encoding="utf-8")))
    edca = list(csv.DictReader((run / "edca_step_runtime.csv").open(encoding="utf-8")))
    native = {"VO": (2, 3, 7, 2080), "VI": (2, 7, 15, 4096),
              "BE": (3, 15, 1023, 0), "BK": (7, 15, 1023, 0)}
    neighbor_bad = 0
    for row in edca:
        actual = tuple(int(float(row[k])) for k in ("aifsn", "cwmin", "cwmax", "txop_us"))
        if int(row["bss"]) > 0 and actual != native[row["ac"]]:
            neighbor_bad += 1
    applied = [r for r in edca if int(r["bss"]) == 0 and abs(float(r["time_s"]) - 1.0) < 1e-6]
    change_by_step = {float(r["time_s"]): int(r["action_changed"]) for r in edca
                      if int(r["bss"]) == 0 and r["role"] == "AP" and r["ac"] == "VO"}
    return {"association_rows": len(associations),
            "association_mismatches": sum(r["match"] != "1" for r in associations),
            "neighbor_rows_checked": sum(int(r["bss"]) > 0 for r in edca),
            "neighbor_nondefault_rows": neighbor_bad,
            "target_rows_at_action": len(applied),
            "requested_profile": action[0], "requested_custom_growth": custom,
            "logged_profiles": sorted(set(int(r["profile"]) for r in applied)),
            "logged_custom_growth": sorted(set(int(r["custom_growth"]) for r in applied)),
            "action_changed_by_step": change_by_step}


def run_one(config: dict, name: str, spec: dict, seed: int, duration: float, root: Path) -> dict:
    path = root / name
    env = TgaxControlEnvironment(config, "default" if not spec["custom"] else "compressed_async_age",
                                 seed, path, duration)
    try:
        env.reset_observation()
        metric_rows = []
        for step in range(round(duration)):
            _, reward, metrics = env.step(np.asarray(spec["action"], dtype=np.int64), 1.0,
                                          custom=spec["custom"])
            assert reward == metrics["reward"]
            metric_rows.append(metrics)
    finally:
        env.close()
    summary = summarize(metric_rows, duration, path, config)
    summary.update(name=name, description=spec["description"], seed=seed, duration_s=duration,
                   action=spec["action"], custom_growth=spec["custom"], audit=audit(path, spec["action"], spec["custom"]))
    (path / "fixed_action_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    step_rows = []
    for index, metrics in enumerate(metric_rows, 1):
        step_rows.append({"step": index, **{k: json.dumps(v) if isinstance(v, (list, dict)) else v
                         for k, v in metrics.items() if not k.endswith("_delay_samples_ms")}})
    save_csv(path / "fixed_action_steps.csv", step_rows)
    return summary


def flatten(summary: dict) -> dict:
    row = {k: v for k, v in summary.items() if not isinstance(v, (list, dict))}
    row["action"] = json.dumps(summary["action"])
    for ac in ACS:
        for direction in DIRECTIONS:
            p = f"{direction}_{ac}"
            for suffix in ("delay_p95_ms", "successful_samples", "throughput_mbps", "final_drop_rate",
                           "retransmission_rate", "risk_packets", "pending_end"):
                row[f"{p}_{suffix}"] = summary[f"{p}_{suffix}"]
        row[f"{ac}_on_time_delivery_rate"] = summary[f"{ac}_on_time_delivery_rate"]
        row[f"{ac}_deadline_eligible"] = summary[f"{ac}_deadline_eligible"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    config = deepcopy(load_config(ROOT / "configs/tgax_ddqn_v1.yaml"))
    root = ROOT / "results/raw" / args.tag
    if root.exists():
        raise SystemExit(f"Refusing to overwrite {root}")
    root.mkdir(parents=True)
    (root / "experiment_config.json").write_text(json.dumps({"seed": args.seed, "duration_s": args.duration,
        "traffic_warmup_s": config["training"]["traffic_warmup_s"], "actions": ACTIONS}, indent=2), encoding="utf-8")
    results = []
    for name, spec in ACTIONS.items():
        print(json.dumps({"state": "starting", "name": name}), flush=True)
        summary = run_one(config, name, spec, args.seed, args.duration, root)
        results.append(summary)
        print(json.dumps({"state": "complete", "name": name, "reward": summary["reward_mean"]}), flush=True)
    demand = [r["application_generated_by_step"] for r in results]
    audit_result = {"paired_generated_demand_equal_each_step": all(x == demand[0] for x in demand[1:]),
                    "generated_demand_by_step": dict(zip(ACTIONS, demand)),
                    "all_associations_correct": all(r["audit"]["association_mismatches"] == 0 for r in results),
                    "all_neighbors_default": all(r["audit"]["neighbor_nondefault_rows"] == 0 for r in results)}
    (root / "paired_audit.json").write_text(json.dumps(audit_result, indent=2), encoding="utf-8")
    (root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_csv(root / "comparison.csv", [flatten(r) for r in results])
    if not all(audit_result.values() if isinstance(audit_result, dict) else []):
        raise RuntimeError(f"Paired fixed-action audit failed: {audit_result}")


if __name__ == "__main__":
    main()
