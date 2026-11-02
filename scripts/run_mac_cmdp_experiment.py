#!/usr/bin/env python3
"""One-YAML, two-command entry point for direct-MAC CMDP experiments.

Usage:
  python scripts/run_mac_cmdp_experiment.py train
  python scripts/run_mac_cmdp_experiment.py plot

``train`` prepares inputs, trains/resumes every seed, runs final-checkpoint
evaluation, and plots. Cost thresholds come directly from YAML; no Default
budget calibration is performed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config

ACS = ("VO", "VI", "BE", "BK")
AGGREGATE_COSTS = ("delay", "loss", "fairness", "starvation")
DIRECTIONAL_COSTS = ("delay_ul", "delay_dl", "loss_ul", "loss_dl", "fairness", "starvation")
TWO_COSTS = ("qos", "starvation")
THREE_COSTS = ("qos_avg", "qos_max", "starvation")
FIVE_COSTS = ("qos_vo_dl", "qos_vo_ul", "qos_vi_dl", "qos_vi_ul", "starvation")
MODELS = {"VOIP", "GMG", "BV3", "BV6", "HTTP", "FTP", "HTTP_REQ"}
REFERENCE_TRANSPORT = {"VOIP": "UDP", "GMG": "UDP", "BV3": "TCP", "BV6": "TCP",
                       "HTTP": "TCP", "FTP": "TCP", "HTTP_REQ": "TCP"}
PRINT_LOCK = threading.Lock()


def read_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Experiment YAML must contain a mapping")
    return data


def canonical_digest(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def require_fraction(value, name: str, *, upper_open: bool = False) -> float:
    result = float(value)
    if result < 0 or result > 1 or (upper_open and result == 1):
        raise ValueError(f"{name} must lie in [0,1{' )' if upper_open else ']'}")
    return result


def validate(spec: dict) -> dict:
    experiment = spec.get("experiment", {})
    exp_id = str(experiment.get("id", ""))
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", exp_id):
        raise ValueError("experiment.id must be 3-64 lowercase letters/digits/underscores and start with a letter")
    for key in ("topology_template", "parent_config"):
        candidate = ROOT / experiment.get(key, "")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
    output_root = spec.get("output", {}).get("root")
    if not isinstance(output_root, str) or not output_root.strip():
        raise ValueError("output.root must be a non-empty path")

    rows = spec.get("traffic")
    if not isinstance(rows, list) or not rows:
        raise ValueError("traffic must be a non-empty list")
    names = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"traffic[{index}] must be a mapping")
        name = str(row.get("name", ""))
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in names:
            raise ValueError(f"traffic[{index}].name must be a unique lowercase identifier")
        names.add(name)
        model = str(row.get("model", "")).upper()
        direction = str(row.get("direction", "")).upper()
        ac = str(row.get("ac", "")).upper()
        if model not in MODELS or direction not in ("UL", "DL") or ac not in ACS:
            raise ValueError(f"Invalid model/direction/ac in traffic item {name}")
        if int(row.get("count", -1)) < 0:
            raise ValueError(f"traffic[{name}].count must be non-negative")
        assignment = str(row.get("assignment", "spread"))
        if assignment not in ("spread", "round_robin", "explicit"):
            raise ValueError(f"Unsupported assignment {assignment} in {name}")
        if assignment == "explicit":
            stas = row.get("stas")
            if not isinstance(stas, list) or not stas or any(int(x) not in range(40) for x in stas):
                raise ValueError(f"traffic[{name}].stas must list STA indices 0..39")

    cmdp = spec.get("cmdp", {})
    weights = cmdp.get("utility_weights", {})
    if set(weights) != set(ACS) or abs(sum(float(weights[a]) for a in ACS) - 1.0) > 1e-12:
        raise ValueError("cmdp.utility_weights must contain VO/VI/BE/BK and sum to one")
    budgets = cmdp.get("cost_budgets", {})
    if set(budgets) in (set(TWO_COSTS), set(THREE_COSTS), set(FIVE_COSTS)):
        cost_names = (FIVE_COSTS if set(budgets) == set(FIVE_COSTS) else
                      THREE_COSTS if set(budgets) == set(THREE_COSTS) else TWO_COSTS)
        references = cmdp.get("qos_delay_reference_ms", {})
        if set(references) != {"VO", "VI"} or any(float(references[ac]) <= 0 for ac in ("VO", "VI")):
            raise ValueError("cmdp.qos_delay_reference_ms must contain positive VO and VI values")
        obsolete = {"delay_reference_ms", "delay_excess_cap", "minimum_jain", "minimum_satisfaction"}
        present = sorted(obsolete & set(cmdp))
        if present:
            raise ValueError(f"QoS CMDP contains obsolete fields: {present}")
    elif set(budgets) == set(AGGREGATE_COSTS):
        cost_names = AGGREGATE_COSTS
    elif set(budgets) == set(DIRECTIONAL_COSTS):
        cost_names = DIRECTIONAL_COSTS
    else:
        raise ValueError("cmdp.cost_budgets must use aggregate, directional, two-Cost, three-Cost, or five-Cost names")
    if cost_names not in (TWO_COSTS, THREE_COSTS, FIVE_COSTS):
        references = cmdp.get("delay_reference_ms", {})
        if set(references) != {"VO", "VI"} or any(float(references[ac]) <= 0 for ac in ("VO", "VI")):
            raise ValueError("cmdp.delay_reference_ms must contain positive VO and VI values")
        if float(cmdp.get("delay_excess_cap", 0)) <= 0:
            raise ValueError("cmdp.delay_excess_cap must be positive")
        minimum_jain = require_fraction(cmdp.get("minimum_jain"), "cmdp.minimum_jain")
        if minimum_jain <= 0:
            raise ValueError("cmdp.minimum_jain must be positive")
        minimum_satisfaction = cmdp.get("minimum_satisfaction", {})
        if set(minimum_satisfaction) != {"BE", "BK"}:
            raise ValueError("cmdp.minimum_satisfaction must contain BE and BK")
        if any(not 0 < float(minimum_satisfaction[ac]) <= 1 for ac in ("BE", "BK")):
            raise ValueError("cmdp.minimum_satisfaction values must lie in (0,1]")
    for name in cost_names:
        require_fraction(budgets[name], f"cmdp.cost_budgets.{name}")
    initial = cmdp.get("multiplier_initial")
    if not isinstance(initial, list) or len(initial) != len(cost_names) or any(float(value) < 0 for value in initial):
        raise ValueError("cmdp.multiplier_initial must contain one non-negative value per Cost")
    if float(cmdp.get("multiplier_learning_rate", 0)) <= 0:
        raise ValueError("cmdp.multiplier_learning_rate must be positive")
    if not 0 < float(cmdp.get("multiplier_ema_alpha", 0)) <= 1:
        raise ValueError("cmdp.multiplier_ema_alpha must lie in (0,1]")
    if float(cmdp.get("multiplier_maximum", 0)) <= 0:
        raise ValueError("cmdp.multiplier_maximum must be positive")

    simulation = spec.get("simulation", {})
    for key in ("training_seeds", "evaluation_seeds"):
        values = simulation.get(key)
        if not isinstance(values, list) or not values or len(set(map(int, values))) != len(values):
            raise ValueError(f"simulation.{key} must be a non-empty unique list")
    overlap = sorted(set(map(int, simulation["training_seeds"])) &
                     set(map(int, simulation["evaluation_seeds"])))
    if overlap:
        raise ValueError(
            "Training and evaluation seeds must be disjoint; repeated seeds: "
            f"{overlap}. Use unseen seeds for final-checkpoint evaluation."
        )
    training = spec.get("training", {})
    for key in ("stages", "updates_per_stage", "epsilon_steps", "batch_size", "replay_capacity", "warmup",
                "parallel_workers"):
        if int(training.get(key, 0)) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if float(training.get("episode_s", 0)) <= 0:
        raise ValueError("training.episode_s must be positive")
    return spec


def paths(spec: dict, output_override: Path | None = None) -> dict[str, Path]:
    root = output_override if output_override is not None else Path(spec["output"]["root"])
    if not root.is_absolute():
        root = ROOT / root
    root = root.resolve()
    return {"root": root, "inputs": root / "inputs", "run": root,
            "summary": root / "summary", "figures": root / "figures",
            "generated": root / "generated_config", "logs": root / "logs"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def assigned_sta(item: dict, index: int, count: int) -> int:
    offset = int(item.get("sta_offset", 0)) % 40
    mode = item.get("assignment", "spread")
    if mode == "explicit":
        values = [int(x) for x in item["stas"]]
        return values[index % len(values)]
    if mode == "round_robin":
        return (offset + index) % 40
    return (offset + (index * 40 // max(1, count))) % 40 if count <= 40 else (offset + index) % 40


def _config_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _inheritance_path(parent: Path, generated_dir: Path) -> str:
    """Prefer portable relative inheritance, but support --output on another drive."""
    try:
        return os.path.relpath(parent, generated_dir).replace("\\", "/")
    except ValueError:
        return str(parent.resolve()).replace("\\", "/")


def prepare(spec: dict, source_yaml: Path, p: dict[str, Path]) -> Path:
    """Generate workload and the inherited JSON config inside output.root."""
    digest = canonical_digest(spec)
    manifest_path = p["inputs"] / "manifest.json"
    train_config = p["generated"] / "train.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("spec_sha256") != digest:
            raise RuntimeError(f"Output contains another YAML configuration: {p['root']}\n"
                               "Choose another output.root or pass --output.")
        if not train_config.is_file():
            raise RuntimeError(f"Generated config is missing: {train_config}")
        return train_config
    if p["inputs"].exists() or p["generated"].exists():
        raise RuntimeError(f"Incomplete generated data exists under {p['root']}; choose another output directory")

    topology = read_csv(ROOT / spec["experiment"]["topology_template"])
    stas = {int(row["sta"]): row for row in topology if row["role"] == "STA"}
    if len(stas) != 40 or set(stas) != set(range(40)):
        raise RuntimeError("Topology must contain STA indices 0..39")
    flow_rows = []
    flow_id = stream_id = 900000
    counts: dict[str, int] = {}
    for item in spec["traffic"]:
        model, direction, ac = item["model"].upper(), item["direction"].upper(), item["ac"].upper()
        count = int(item["count"])
        key = f"{model}/{direction}/{ac}"
        counts[key] = counts.get(key, 0) + count
        for index in range(count):
            sid = assigned_sta(item, index, count)
            label = stas[sid]["node_label"]
            flow_rows.append({"flow_id": str(flow_id), "stream_id": str(stream_id), "bss": "0",
                              "service": f"USER_{item['name'].upper()}_{index}", "model": model,
                              "assignment": str(index), "direction": direction, "transport": "NONE", "ac": ac,
                              "sta": str(sid), "source": label if direction == "UL" else "ap0",
                              "sink": "ap0" if direction == "UL" else label,
                              "reference_transport": REFERENCE_TRANSPORT[model]})
            flow_id += 1
            stream_id += 1
    fields = ["flow_id", "stream_id", "bss", "service", "model", "assignment", "direction",
              "transport", "ac", "sta", "source", "sink", "reference_transport"]
    write_csv(p["inputs"] / "topology.csv", topology, list(topology[0]))
    write_csv(p["inputs"] / "flows.csv", flow_rows, fields)
    p["generated"].mkdir(parents=True)

    cmdp, sim, train = spec["cmdp"], spec["simulation"], spec["training"]
    weights = cmdp["utility_weights"]
    budget_keys=set(cmdp["cost_budgets"])
    cost_names = (FIVE_COSTS if budget_keys == set(FIVE_COSTS) else
                   THREE_COSTS if budget_keys == set(THREE_COSTS) else
                  TWO_COSTS if budget_keys == set(TWO_COSTS) else
                  DIRECTIONAL_COSTS if budget_keys == set(DIRECTIONAL_COSTS) else AGGREGATE_COSTS)
    budgets = [float(cmdp["cost_budgets"][name]) for name in cost_names]
    parent_path = ROOT / spec["experiment"]["parent_config"]
    # Materialize the inherited base and replace the reward block completely.
    # This prevents unused legacy Cost fields from surviving dictionary merges.
    generated = load_config(parent_path)
    generated["identity"] = {"environment": spec["experiment"]["id"],
                             "purpose": spec["experiment"].get("description", "one-YAML CMDP experiment")}
    generated["backend"].update(topology=_config_path(p["inputs"] / "topology.csv"),
                                flows=_config_path(p["inputs"] / "flows.csv"),
                                direct_high_watermark=int(sim.get("direct_high_watermark", 0)))
    generated["reward"] = {
        "version": ("tgax_cmdp_five_cost_v6" if cost_names == FIVE_COSTS else
                    "tgax_cmdp_three_cost_v5" if cost_names == THREE_COSTS else
                    "tgax_cmdp_two_cost_v4" if cost_names == TWO_COSTS else
                    "tgax_cmdp_directional_v3" if cost_names == DIRECTIONAL_COSTS else "tgax_cmdp_v2"),
        "cost_order": list(cost_names),
        "utility_weights": [float(weights[a]) for a in ACS],
        "budgets": budgets, "budget_status": "manual; no Default calibration",
        "budget_source": _config_path(source_yaml),
        "multiplier_initial": list(map(float, cmdp["multiplier_initial"])),
        "multiplier_learning_rate": float(cmdp["multiplier_learning_rate"]),
        "multiplier_ema_alpha": float(cmdp["multiplier_ema_alpha"]),
        "multiplier_maximum": float(cmdp["multiplier_maximum"])}
    if cost_names in (TWO_COSTS, THREE_COSTS, FIVE_COSTS):
        generated["reward"].update(
            qos_delay_reference_ms=[float(cmdp["qos_delay_reference_ms"]["VO"]),
                                    float(cmdp["qos_delay_reference_ms"]["VI"])],
            qos_threshold_ms=[float(cmdp["qos_delay_reference_ms"]["VO"]),
                              float(cmdp["qos_delay_reference_ms"]["VI"])])
    else:
        generated["reward"].update(
            qos_threshold_ms=[float(cmdp["delay_reference_ms"]["VO"]),
                              float(cmdp["delay_reference_ms"]["VI"])],
            delay_reference_ms=[float(cmdp["delay_reference_ms"]["VO"]),
                                float(cmdp["delay_reference_ms"]["VI"])],
            delay_excess_cap=float(cmdp["delay_excess_cap"]),
            minimum_jain=float(cmdp["minimum_jain"]),
            minimum_satisfaction=[float(cmdp["minimum_satisfaction"]["BE"]),
                                  float(cmdp["minimum_satisfaction"]["BK"])])
    generated["training"].update(train)
    generated["run"] = {"mode": spec["experiment"]["id"],
                        "seeds": list(map(int, sim["training_seeds"])),
                        "workers": int(train.get("parallel_workers", 1)),
                        "output": _config_path(p["root"])}
    generated["evaluation"] = {"duration_s": float(sim["evaluation_duration_s"]), "seeds": [],
                               "held_out_seeds": list(map(int, sim["evaluation_seeds"])),
                               "methods": ["default", "compressed_async_age"],
                               "every_training_steps": 0, "final_only": True,
                               "checkpoint_metric": "final_checkpoint_only"}
    train_config.write_text(json.dumps(generated, indent=2, ensure_ascii=False), encoding="utf-8")
    load_config(train_config)
    manifest = {"experiment_id": spec["experiment"]["id"], "spec": _config_path(source_yaml),
                "spec_sha256": digest, "flow_count": len(flow_rows), "counts": counts,
                "cost_order": list(cost_names), "cost_budgets": budgets,
                "qos_delay_reference_ms": cmdp.get("qos_delay_reference_ms"),
                "note": "Cost budgets are manual. No Default budget calibration was run."}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return train_config


def run_command(args: list[str], log: Path, *, prefix: str = "") -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            with PRINT_LOCK:
                print(f"{prefix}{line}", end="", flush=True)
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f"Command failed ({returncode}); see {log}")


def train_one(seed: int, expected: int, train_config: Path, p: dict[str, Path]) -> None:
    seed_root = p["root"] / f"seed_{seed}"
    status = seed_root / "status.json"
    if status.is_file() and json.loads(status.read_text(encoding="utf-8")).get("state") == "complete":
        with PRINT_LOCK:
            print(f"Seed {seed} already complete; skipping.", flush=True)
        return
    snapshot = seed_root / "snapshots" / f"step_{expected:03d}.pt"
    checkpoint = seed_root / "train" / "checkpoint.pt"
    if seed_root.exists() and not checkpoint.is_file() and not snapshot.is_file():
        interrupted_root = p["root"] / "interrupted"
        interrupted_root.mkdir(exist_ok=True)
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = interrupted_root / f"seed_{seed}_{suffix}"
        counter = 1
        while destination.exists():
            destination = interrupted_root / f"seed_{seed}_{suffix}_{counter}"
            counter += 1
        shutil.move(str(seed_root), str(destination))
        with PRINT_LOCK:
            print(f"Preserved incomplete pre-checkpoint run: {destination}", flush=True)
    if snapshot.is_file():
        args = [sys.executable, str(ROOT / "scripts" / "finalize_tgax_ddqn_final_only.py"),
                "--config", str(train_config), "--root", str(seed_root)]
    else:
        # The trainer's short Default prepass only fits the observation
        # scaler. It never calculates or changes CMDP Cost thresholds.
        args = [sys.executable, str(ROOT / "scripts" / "run_tgax_ddqn_extended.py"),
                "--config", str(train_config), "--root", str(seed_root), "--seed", str(seed)]
        if checkpoint.is_file():
            args.append("--resume")
    with PRINT_LOCK:
        print(f"Starting seed {seed}; output={seed_root}", flush=True)
    run_command(args, p["logs"] / f"seed_{seed}_train.log", prefix=f"[seed {seed}] ")


def train_all(spec: dict, train_config: Path, p: dict[str, Path]) -> None:
    config = load_config(train_config)
    expected = int(config["training"]["stages"] * config["training"]["episode_s"] /
                   config["timing"]["decision_s"])
    seeds = list(map(int, spec["simulation"]["training_seeds"]))
    workers = min(int(spec["training"].get("parallel_workers", 1)), len(seeds))
    print(f"Seed execution: {len(seeds)} seed(s), {workers} parallel worker(s)", flush=True)
    if workers == 1:
        for seed in seeds:
            train_one(seed, expected, train_config, p)
        return
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers,
                                               thread_name_prefix="seed-worker") as executor:
        futures = {executor.submit(train_one, seed, expected, train_config, p): seed for seed in seeds}
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            try:
                future.result()
            except BaseException as exc:
                failures.append((seed, exc))
                with PRINT_LOCK:
                    print(f"Seed {seed} failed: {exc}", file=sys.stderr, flush=True)
    if failures:
        detail = "; ".join(f"seed {seed}: {exc}" for seed, exc in failures)
        raise RuntimeError(f"One or more independent seed jobs failed: {detail}")


def plot_all(spec: dict, p: dict[str, Path]) -> None:
    sim = spec["simulation"]
    configured_seeds = list(map(int, sim["training_seeds"]))
    completed_seeds = []
    for seed in configured_seeds:
        status_path = p["root"] / f"seed_{seed}" / "status.json"
        if status_path.is_file() and json.loads(status_path.read_text(encoding="utf-8")).get("state") == "complete":
            completed_seeds.append(seed)
    if not completed_seeds:
        raise RuntimeError("No completed training seed with final held-out evaluation is available to plot")
    if completed_seeds != configured_seeds:
        print(f"Plotting completed training seeds only: {completed_seeds}", flush=True)
    run_command([sys.executable, str(ROOT / "scripts" / "analyze_mac_cmdp_experiment.py"),
                 "--root", str(p["root"]), "--summary", str(p["summary"]),
                 "--figures", str(p["figures"]), "--train-seeds", *map(str, completed_seeds),
                 "--eval-seeds", *map(str, sim["evaluation_seeds"])], p["logs"] / "analysis.log")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "plot"), help="train/resume and plot, or only re-plot")
    parser.add_argument("--config", type=Path, default=Path("configs/mac_cmdp_experiment.yaml"),
                        help="single YAML (default: configs/mac_cmdp_experiment.yaml)")
    parser.add_argument("--output", type=Path, help="override output.root")
    args = parser.parse_args()
    source = args.config if args.config.is_absolute() else ROOT / args.config
    spec = validate(read_yaml(source))
    p = paths(spec, args.output)
    if args.command == "plot":
        plot_all(spec, p)
        print(f"Figures: {p['figures']}")
        return

    p["root"].mkdir(parents=True, exist_ok=True)
    status_path = p["root"] / "pipeline_status.json"
    status_path.write_text(json.dumps({"state": "running", "config": str(source), "output": str(p["root"])},
                                      indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        train_config = prepare(spec, source, p)
        train_all(spec, train_config, p)
        plot_all(spec, p)
    except Exception as exc:
        status_path.write_text(json.dumps({"state": "failed", "config": str(source),
                                           "output": str(p["root"]), "error": str(exc)},
                                          indent=2, ensure_ascii=False), encoding="utf-8")
        raise
    status_path.write_text(json.dumps({"state": "complete", "config": str(source), "output": str(p["root"]),
                                       "summary": str(p["summary"]), "figures": str(p["figures"])},
                                      indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Experiment complete: {p['root']}")


if __name__ == "__main__":
    main()
