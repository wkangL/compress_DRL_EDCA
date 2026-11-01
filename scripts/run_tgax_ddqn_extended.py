#!/usr/bin/env python3
"""400-step single-seed TGax DDQN run with isolated checkpoint validation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import time
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config
from models.branching_ddqn import ControlNetwork
from reporting.sta_state import StateScaler
from rl.artifacts import atomic_output
from rl.replay_buffer import ReplayBuffer, Transition
from rl.trainer import InferenceCache, explore, freeze_target, set_seed, td_update, update_target
from rl.tgax_environment import TgaxControlEnvironment
from rl.cmdp_reward import LagrangeController
from scripts.run_tgax_ddqn_v1 import baseline_scaler, compact, save_csv
from scripts.run_tgax_fixed_actions import summarize


CONFIG = ROOT / "configs/tgax_ddqn_extended_seed41.yaml"
DEFAULT_ROOT = ROOT / "results/raw/tgax_ddqn_extended_seed41_400step_20261025"
METHOD = "compressed_async_age"
SCENARIO = "candidate_voip8_gaming4_ftp_bk"
BRANCH_NAMES = ("profile", "alpha", "beta1", "beta2", "gamma")


def source_hash(config_path: Path) -> str:
    paths = [Path(__file__), ROOT / "rl/tgax_environment.py", ROOT / "rl/tgax_reward.py",
             ROOT / "rl/cmdp_reward.py",
             ROOT / "rl/trainer.py", ROOT / "models/branching_ddqn.py",
             ROOT / "ns3_ext/tgax-control.cc", config_path]
    return hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()


def save_checkpoint(path: Path, payload: dict) -> None:
    with atomic_output(path, binary=True) as stream:
        torch.save(payload, stream)


def save_live_diagnostics(root: Path, step_rows: list[dict], losses: list[dict]) -> None:
    """Refresh a lightweight PNG after each episode without affecting training."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def smooth(values: list[float], window: int = 20) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if len(data) < 2:
            return data
        width = min(window, len(data))
        return np.convolve(data, np.ones(width) / width, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    steps = [int(row["step"]) for row in step_rows]
    rewards = [float(row["reward"]) for row in step_rows]
    axes[0].plot(steps, rewards, color="#9ecae1", lw=0.8, alpha=0.75, label="per step")
    reward_mean = smooth(rewards)
    if len(reward_mean):
        axes[0].plot(steps[len(steps) - len(reward_mean):], reward_mean,
                     color="#08519c", lw=2.0, label="moving mean")
    axes[0].set(title="Training reward (live)", xlabel="environment step", ylabel="Lagrangian reward")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    if losses:
        updates = [int(row["update"]) for row in losses]
        qloss = [float(row["td_loss"]) for row in losses]
        axes[1].plot(updates, qloss, color="#fcae91", lw=0.8, alpha=0.65, label="Q-loss")
        loss_mean = smooth(qloss)
        axes[1].plot(updates[len(updates) - len(loss_mean):], loss_mean,
                     color="#cb181d", lw=2.0, label="moving mean")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "Waiting for replay warmup / first gradient update",
                     ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set(title="Q-loss (live)", xlabel="gradient update", ylabel="Huber loss")
    axes[1].grid(alpha=0.25)
    target = root / "live_training_diagnostics.png"
    temporary = root / "live_training_diagnostics.tmp.png"
    fig.savefig(temporary, dpi=160)
    plt.close(fig)
    temporary.replace(target)


def action_histogram(rows: list[dict]) -> list[dict]:
    counters = [Counter() for _ in BRANCH_NAMES]
    for row in rows:
        action = row["action"] if isinstance(row["action"], list) else json.loads(row["action"])
        for i, value in enumerate(action):
            counters[i][int(value)] += 1
    return [{"branch": name, "counts": dict(sorted(counter.items()))}
            for name, counter in zip(BRANCH_NAMES, counters)]


def evaluate_one(config: dict, model: ControlNetwork, scaler: StateScaler, seed: int,
                 directory: Path, learned: bool) -> tuple[dict, list[dict]]:
    duration = float(config["evaluation"]["duration_s"])
    env = TgaxControlEnvironment(config, METHOD if learned else "default", seed, directory, duration)
    rows = []
    inference = InferenceCache(model, scaler) if learned else None
    obs = env.reset_observation()
    try:
        for step in range(round(duration / config["timing"]["decision_s"])):
            action = (inference.select(obs) if learned else
                      np.asarray(config["action"].get("default_action", [0, 0, 5, 5, 2]), dtype=np.int64))
            obs, reward, metrics = env.step(action, config["timing"]["decision_s"], custom=learned)
            rows.append({"step": step + 1, "action": action.tolist(), "reward": reward, **metrics})
    finally:
        env.close()
    summary = summarize(rows, duration, directory, config)
    summary.update(eval_seed=seed, learned=learned, action_distribution=action_histogram(rows),
                   actions=[row["action"] for row in rows])
    (directory / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_csv(directory / "evaluation_steps.csv",
             [{"step": row["step"], "action": json.dumps(row["action"]), **compact(row)} for row in rows])
    return summary, rows


def snapshot_payload(model: ControlNetwork, target: ControlNetwork, scaler: StateScaler,
                     config: dict, stage: int, steps: int, updates: int, version: int,
                     seed: int, digest: str) -> dict:
    return {"model": model.state_dict(), "target": target.state_dict(), "scaler_mean": scaler.mean,
            "scaler_std": scaler.std, "scaler_clip": scaler.delay_clip_ms, "config": config,
            "seed": seed, "stage": stage, "steps": steps, "updates": updates,
            "encoder_version": version, "method": METHOD, "source_hash": digest}


def load_snapshot_model(path: Path, config: dict) -> tuple[ControlNetwork, StateScaler]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = ControlNetwork(config, METHOD, 10)
    model.load_state_dict(saved["model"]); model.eval()
    scaler = StateScaler(saved["scaler_clip"])
    scaler.mean = saved["scaler_mean"]; scaler.std = saved["scaler_std"]; scaler.fitted = True
    return model, scaler


def validation_guard(default: dict, learned: dict) -> tuple[bool, list[str]]:
    """Predeclared guard against selecting reward-only starvation policies."""
    reasons = []
    default_jain = default.get("jain_demand_satisfaction")
    learned_jain = learned.get("jain_demand_satisfaction")
    if default_jain and learned_jain is not None and learned_jain < 0.9 * default_jain:
        reasons.append("Jain fairness below 90% of paired Default")
    for ac in ("vo", "vi"):
        base = default.get(f"{ac}_on_time_delivery_rate")
        trial = learned.get(f"{ac}_on_time_delivery_rate")
        if base is not None and trial is not None and float(trial) < 0.95 * float(base):
            reasons.append(f"{ac.upper()} timely delivery below 95% of paired Default")
    total_base=total_trial=0.
    for ac in ("vo", "vi", "be", "bk"):
        base = sum(float(default.get(f"{direction}_{ac}_throughput_mbps") or 0.)
                   for direction in ("ul", "dl"))
        trial = sum(float(learned.get(f"{direction}_{ac}_throughput_mbps") or 0.)
                    for direction in ("ul", "dl"))
        total_base+=base; total_trial+=trial
        floor = 0.8 if ac in {"be", "bk"} else 0.0
        if base > 1e-9 and trial < floor * base:
            reasons.append(f"{ac.upper()} throughput below {floor:.0%} of paired Default")
        if ac in {"be","bk"}:
            base_sat=default.get(f"{ac}_demand_satisfaction")
            trial_sat=learned.get(f"{ac}_demand_satisfaction")
            if base_sat is not None and trial_sat is not None and float(trial_sat)<0.8*float(base_sat):
                reasons.append(f"{ac.upper()} demand satisfaction below 80% of paired Default")
    if total_base>1e-9 and total_trial<0.95*total_base:
        reasons.append("total successful payload below 95% of paired Default")
    def failure(summary: dict) -> float:
        failed = risk = 0.
        for ac in ("vo", "vi", "be", "bk"):
            for direction in ("ul", "dl"):
                prefix = f"{direction}_{ac}"
                count = float(summary.get(f"{prefix}_risk_packets") or 0.)
                rate = summary.get(f"{prefix}_final_drop_rate")
                if rate is not None:
                    failed += count * float(rate)
                risk += count
        return failed / risk if risk else 0.
    if failure(learned) > failure(default) + 0.05:
        reasons.append("aggregate final failure rate exceeds Default by >5 percentage points")
    return not reasons, reasons


def validate_checkpoint(config: dict, root: Path, snapshot: Path, stage: int,
                        model: ControlNetwork, scaler: StateScaler, steps: int,
                        updates: int) -> list[dict]:
    records = []
    for seed in config["evaluation"]["seeds"]:
        baseline_dir = root / "validation" / "baseline" / f"seed_{seed}"
        if (baseline_dir / "evaluation_summary.json").is_file():
            baseline = json.loads((baseline_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
        else:
            baseline, _ = evaluate_one(config, model, scaler, seed, baseline_dir, False)
        learned_dir = root / "validation" / f"step_{steps:03d}" / f"seed_{seed}"
        learned, _ = evaluate_one(config, model, scaler, seed, learned_dir, True)
        paired = baseline["application_generated_by_step"] == learned["application_generated_by_step"]
        guard, reasons = validation_guard(baseline, learned)
        records.append({"stage": stage, "training_steps": steps, "gradient_updates": updates,
                        "eval_seed": seed, "default_reward": baseline["reward_mean"],
                        "learned_reward": learned["reward_mean"],
                        "reward_change_percent": (learned["reward_mean"] / baseline["reward_mean"] - 1) * 100,
                        "paired_generated_demand_equal_each_step": paired,
                        "safety_guard_pass": guard, "safety_guard_reasons": json.dumps(reasons),
                        "snapshot": str(snapshot.relative_to(root)),
                        "learned_cmdp_utility": learned.get("cmdp_utility_mean"),
                        **{f"learned_{k}": learned.get(k) for k in
                           ("jain_demand_satisfaction", "vo_on_time_delivery_rate", "vi_on_time_delivery_rate")}})
    return records


def training_checkpoint(model, target, optimizer, scaler, replay, rng, config, stage,
                        steps, updates, version, step_rows, losses, validation_rows,
                        seed, digest, lagrange=None):
    payload={**snapshot_payload(model, target, scaler, config, stage, steps, updates, version, seed, digest),
            "optimizer": optimizer.state_dict(), "replay": list(replay.data),
            "replay_rng": replay.rng.bit_generator.state, "rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(), "training_steps": step_rows,
            "losses": losses, "validation_rows": validation_rows}
    if lagrange is not None:
        payload["lagrange"]={"budgets":lagrange.budgets,"multipliers":lagrange.multipliers,
            "ema_costs":lagrange.ema_costs,"learning_rate":lagrange.learning_rate,
            "ema_alpha":lagrange.ema_alpha,"maximum":lagrange.maximum}
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scaler-events", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path=args.config.resolve(); config = load_config(config_path)
    seed = args.seed if args.seed is not None else config["run"]["seeds"][0]
    root = (args.root or DEFAULT_ROOT).resolve()
    scenario=config.get("identity",{}).get("environment",SCENARIO)
    digest=source_hash(config_path)
    tc = config["training"]
    cmdp_version=config["reward"].get("version")
    is_cmdp=cmdp_version in ("tgax_cmdp_v1", "tgax_cmdp_v2")
    lagrange=None
    if is_cmdp:
        if config["reward"].get("budgets") is None:
            raise ValueError("CMDP training requires four explicitly configured cost budgets")
        reward=config["reward"]
        lagrange=LagrangeController(np.asarray(reward["budgets"],dtype=float),
            np.asarray(reward["multiplier_initial"],dtype=float),reward["multiplier_learning_rate"],
            reward["multiplier_ema_alpha"],reward["multiplier_maximum"])
    steps_per_episode = int(round(tc["episode_s"] / config["timing"]["decision_s"]))
    expected_steps = int(tc["stages"] * steps_per_episode)
    if steps_per_episode <= 0 or tc["updates_per_stage"] != steps_per_episode:
        raise ValueError("Training requires one gradient update per new transition after warmup")
    if root.exists() and not args.resume:
        raise SystemExit(f"Refusing to overwrite {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    first_update_after = int(math.ceil(tc["warmup"] / steps_per_episode) * steps_per_episode)
    updated_episodes = max(0, tc["stages"] - first_update_after // steps_per_episode + 1)
    total_updates = updated_episodes * tc["updates_per_stage"]
    budget = {"environment_steps": expected_steps, "episodes": tc["stages"],
              "steps_per_episode": steps_per_episode,
              "warmup_transitions": tc["warmup"], "first_update_after_steps": first_update_after,
              "updates_per_updated_episode": tc["updates_per_stage"],
              "total_gradient_updates": total_updates,
              "new_transition_to_update_ratio_after_warmup": 1.0,
              "overall_update_to_transition_ratio": total_updates / expected_steps,
              "episode_update_semantics": "collect 20 with frozen parameters, then perform 20 updates"}
    (root / "budget.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")

    set_seed(seed, tc["threads"]); rng = np.random.default_rng(seed)
    if args.resume:
        scaler = StateScaler(config["state"]["delay_clip_ms"])
    elif args.scaler_events is not None:
        scaler = baseline_scaler(config, args.scaler_events.resolve())
    else:
        scaler_dir=root/"scaler_default"
        print(json.dumps({"phase": "scaler_default_start", "seed": 40,
                          "duration_s": 5.0,
                          "note": "state normalization only; CMDP budgets are unchanged"}), flush=True)
        scaler_env=TgaxControlEnvironment(config,"default",40,scaler_dir,5.0,0)
        scaler_obs=scaler_env.reset_observation()
        try:
            default_action=np.asarray(config["action"].get("default_action",[0,0,5,5,2]),dtype=np.int64)
            for _ in range(5):
                scaler_obs,_,_=scaler_env.step(default_action,1.0,custom=False)
        finally:
            scaler_env.close()
        scaler=baseline_scaler(config,scaler_dir/"events.log")
        print(json.dumps({"phase": "scaler_default_complete", "seed": 40,
                          "duration_s": 5.0}), flush=True)
    model = ControlNetwork(config, METHOD, 10); target = freeze_target(deepcopy(model))
    optimizer = torch.optim.AdamW([
        {"params": model.q.parameters(), "lr": tc["q_lr"]},
        {"params": list(model.encoder.parameters()) + list(model.aggregator.parameters()),
         "lr": tc["representation_lr"]}])
    replay = ReplayBuffer(tc["replay_capacity"], seed)
    steps = updates = version = 0; first_stage = 0
    step_rows: list[dict] = []; losses: list[dict] = []; validation_rows: list[dict] = []
    checkpoint_path = root / "train" / "checkpoint.pt"; checkpoint_path.parent.mkdir(exist_ok=True)
    if args.resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["config"] != config or saved["source_hash"] != digest or saved["seed"] != seed:
            raise ValueError("Resume configuration or source hash mismatch")
        model.load_state_dict(saved["model"]); target.load_state_dict(saved["target"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.mean = saved["scaler_mean"]; scaler.std = saved["scaler_std"]; scaler.fitted = True
        replay.data.extend(saved["replay"]); replay.rng.bit_generator.state = saved["replay_rng"]
        rng.bit_generator.state = saved["rng"]; torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"]); random.setstate(saved["python_rng"])
        steps, updates, version = saved["steps"], saved["updates"], saved["encoder_version"]
        first_stage = saved["stage"]; step_rows = saved["training_steps"]
        losses = saved["losses"]; validation_rows = saved["validation_rows"]
        if is_cmdp:
            state=saved["lagrange"]; lagrange.multipliers=np.asarray(state["multipliers"],dtype=float)
            lagrange.ema_costs=(None if state["ema_costs"] is None else np.asarray(state["ema_costs"],dtype=float))

    started = time.perf_counter()
    try:
        for stage0 in range(first_stage, tc["stages"]):
            stage = stage0 + 1
            env_seed = seed * 100 + stage
            env = TgaxControlEnvironment(config, METHOD, env_seed, root / "train" / f"episode_{stage:02d}",
                                         tc["episode_s"], version)
            if is_cmdp: env.lagrange=lagrange
            obs = env.reset_observation(); inference = InferenceCache(model, scaler)
            episode_costs=[]; episode_cost_valid=[]; episode_rewards=[]
            try:
                for local in range(steps_per_episode):
                    epsilon = tc["epsilon_end"] + (tc["epsilon_start"] - tc["epsilon_end"]) * max(0, 1 - steps / tc["epsilon_steps"])
                    exploratory = rng.random() < epsilon
                    action = explore(model, rng) if exploratory else inference.select(obs)
                    following, reward, metrics = env.step(action, 1.0, custom=True)
                    replay.add(Transition(obs, action, reward, following, False, version,
                                          local == steps_per_episode - 1,
                                          scenario, exploratory,
                                          utility=(metrics.get("cmdp_utility") if is_cmdp else None),
                                          costs=(np.asarray(metrics["cmdp_costs"],dtype=float) if is_cmdp else None)))
                    if is_cmdp:
                        episode_costs.append(np.asarray(metrics["cmdp_costs"],dtype=float))
                        if cmdp_version == "tgax_cmdp_v2":
                            episode_cost_valid.append(np.asarray(metrics["cmdp_cost_valid"], dtype=bool))
                        else:
                            episode_cost_valid.append(np.asarray([
                                metrics["vo_on_time_delivery_valid"],
                                metrics["vi_on_time_delivery_valid"], True, True], dtype=bool))
                    obs = following; steps += 1
                    episode_rewards.append(float(reward))
                    step_rows.append({"episode": stage, "episode_step": local + 1, "step": steps,
                                      "environment_seed": env_seed, "epsilon": epsilon,
                                      "exploratory": exploratory, "action": action.tolist(), **compact(metrics)})
                    print(json.dumps({"phase": "train", "episode": stage, "step": steps,
                                      "epsilon": epsilon, "reward": reward}), flush=True)
            finally:
                env.close()
            if is_cmdp:
                costs_array=np.asarray(episode_costs); valid_array=np.asarray(episode_cost_valid)
                valid_mask=valid_array.any(axis=0)
                episode_mean=np.asarray([costs_array[valid_array[:,i],i].mean()
                    if valid_mask[i] else lagrange.budgets[i] for i in range(4)])
                lagrange.update(episode_mean,valid_mask)
            if len(replay) >= tc["warmup"]:
                model.train()
                for _ in range(tc["updates_per_stage"]):
                    stats = td_update(model, target, replay.sample(tc["batch_size"]), scaler, optimizer, tc,
                                      ({"budgets":lagrange.budgets,"multipliers":lagrange.multipliers,
                                        "excess_only":cmdp_version == "tgax_cmdp_v1"}
                                       if is_cmdp else None))
                    if is_cmdp:
                        cost_names = (("delay", "loss", "fairness", "starvation")
                                      if cmdp_version == "tgax_cmdp_v2"
                                      else ("vo", "vi", "jain", "low"))
                        stats.update(**{f"lambda_{name}":float(lagrange.multipliers[i])
                            for i,name in enumerate(cost_names)})
                    updates += 1
                    synced, before, after = update_target(model, target, updates, tc)
                    stats.update(episode=stage, training_steps=steps, update=updates,
                                 target_sync=synced, target_distance_before=before,
                                 target_distance_after=after)
                    losses.append(stats)
                    numeric = [stats[k] for k in ("td_loss", "q_min", "q_max", "target_q_min",
                                                   "target_q_max", "td_error_max_abs", "grad_norm")]
                    if not all(math.isfinite(v) for v in numeric):
                        raise FloatingPointError("Nonfinite TGax training diagnostic")
                    if max(abs(stats["q_min"]), abs(stats["q_max"]), abs(stats["target_q_min"]),
                           abs(stats["target_q_max"])) > 1e4 or stats["grad_norm"] > 1e3:
                        raise FloatingPointError("TGax training divergence threshold exceeded")
                version += 1; model.eval()
            save_csv(root / "training_steps.csv", step_rows); save_csv(root / "losses.csv", losses)
            save_live_diagnostics(root, step_rows, losses)
            payload = training_checkpoint(model, target, optimizer, scaler, replay, rng, config, stage,
                                          steps, updates, version, step_rows, losses, validation_rows,
                                          seed, digest, lagrange)
            save_checkpoint(checkpoint_path, payload)
            evaluation_interval = int(config["evaluation"].get("every_training_steps", 0))
            if evaluation_interval > 0 and steps % evaluation_interval == 0:
                snapshot = root / "snapshots" / f"step_{steps:03d}.pt"; snapshot.parent.mkdir(exist_ok=True)
                save_checkpoint(snapshot, snapshot_payload(model, target, scaler, config, stage, steps, updates, version,
                                                            seed, digest))
                records = validate_checkpoint(config, root, snapshot, stage, model, scaler, steps, updates)
                validation_rows.extend(records); save_csv(root / "validation" / "learning_curve.csv", validation_rows)
                # Commit validation rows into the resumable checkpoint.
                payload["validation_rows"] = validation_rows
                save_checkpoint(checkpoint_path, payload)
            status = {"state": "training", "completed_episodes": stage,
                      "total_episodes": tc["stages"],
                      "steps": steps, "updates": updates, "replay_size": len(replay),
                      "runtime_s": time.perf_counter() - started, "checkpoint": str(checkpoint_path)}
            status.update(progress_percent=100.0 * stage / tc["stages"],
                          episode_reward_mean=float(np.mean(episode_rewards)),
                          epsilon=float(epsilon))
            if losses and len(replay) >= tc["warmup"]:
                recent = losses[-tc["updates_per_stage"]:]
                status.update(q_loss_mean=float(np.mean([row["td_loss"] for row in recent])),
                              q_loss_last=float(recent[-1]["td_loss"]),
                              q_min=float(recent[-1]["q_min"]), q_max=float(recent[-1]["q_max"]),
                              grad_norm=float(recent[-1]["grad_norm"]))
            if is_cmdp:
                status["cmdp_multipliers"] = lagrange.multipliers.tolist()
                status["cmdp_cost_order"] = config["reward"].get(
                    "cost_order", ["vo", "vi", "jain", "low_priority"])
            (root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(json.dumps({"phase": "episode_complete", **status}), flush=True)

        final_snapshot = root / "snapshots" / f"step_{expected_steps:03d}.pt"
        final_snapshot.parent.mkdir(exist_ok=True)
        save_checkpoint(final_snapshot, snapshot_payload(
            model, target, scaler, config, tc["stages"], steps, updates, version, seed, digest))

        grouped = {}; guarded = {}
        for row in validation_rows:
            score=(row.get("learned_cmdp_utility") if is_cmdp else row["learned_reward"])
            grouped.setdefault(int(row["training_steps"]), []).append(float(score))
            guarded.setdefault(int(row["training_steps"]), []).append(bool(row["safety_guard_pass"]))
        eligible=[step for step in sorted(grouped) if all(guarded[step])]
        final_only = bool(config["evaluation"].get("final_only", False))
        if final_only or not grouped:
            best_step = expected_steps
            best_snapshot = final_snapshot
        else:
            best_step = max(eligible or sorted(grouped), key=lambda step: np.mean(grouped[step]))
            best_snapshot = root / "snapshots" / f"step_{best_step:03d}.pt"
        selection = {"metric": ("highest mean CMDP utility among checkpoints passing timely-delivery, "
                                  "payload, BE/BK and fairness guards" if is_cmdp else
                                  "highest mean Reward-v3 among checkpoints passing timely-delivery, throughput and fairness guards"),
                     "mean_reward_by_step": {str(k): float(np.mean(v)) for k, v in sorted(grouped.items())},
                     "eligible_steps": eligible, "selected_step": best_step,
                     "selected_passed_guard": (None if final_only else best_step in eligible),
                     "final_only_evaluation": final_only,
                     "selected_snapshot": str(best_snapshot.relative_to(root)),
                     "final_snapshot": str(final_snapshot.relative_to(root))}
        (root / "validation").mkdir(parents=True, exist_ok=True)
        (root / "validation" / "checkpoint_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

        held_rows = []
        for held_seed in config["evaluation"]["held_out_seeds"]:
            final_model, final_scaler = load_snapshot_model(final_snapshot, config)
            default, _ = evaluate_one(config, final_model, final_scaler, held_seed,
                                      root / "held_out" / f"seed_{held_seed}" / "default", False)
            policies = (("final", final_snapshot),) if final_only else (
                ("selected", best_snapshot), ("final", final_snapshot))
            for label, snapshot in policies:
                eval_model, eval_scaler = load_snapshot_model(snapshot, config)
                summary, _ = evaluate_one(config, eval_model, eval_scaler, held_seed,
                                          root / "held_out" / f"seed_{held_seed}" / label, True)
                held_rows.append({"eval_seed": held_seed, "policy": label,
                                  "checkpoint_step": best_step if label == "selected" else expected_steps,
                                  "default_reward": default["reward_mean"],
                                  "learned_reward": summary["reward_mean"],
                                  "reward_change_percent": (summary["reward_mean"] / default["reward_mean"] - 1) * 100,
                                  "paired_generated_demand_equal_each_step":
                                      default["application_generated_by_step"] == summary["application_generated_by_step"]})
        save_csv(root / "held_out" / "comparison.csv", held_rows)
        final_status = {"state": "complete", "steps": steps, "updates": updates,
                        "selected_step": best_step, "held_out_seeds": config["evaluation"]["held_out_seeds"],
                        "runtime_s": time.perf_counter() - started}
        (root / "status.json").write_text(json.dumps(final_status, indent=2), encoding="utf-8")
        (root / "action_distribution.json").write_text(json.dumps(action_histogram(step_rows), indent=2), encoding="utf-8")
        print(json.dumps({"phase": "complete", **final_status}), flush=True)
    except BaseException as error:
        diagnostic = {"state": "aborted", "error": repr(error), "steps": steps, "updates": updates,
                      "last_losses": losses[-20:], "runtime_s": time.perf_counter() - started}
        (root / "diagnostic_abort.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
        (root / "status.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
