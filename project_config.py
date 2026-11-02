"""Resolve inherited experiment configurations relative to their own file."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def load_config(path: str | Path, _seen: frozenset[Path] = frozenset()) -> dict:
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError("Cyclic configuration inheritance")
    text = path.read_text(encoding="utf-8")
    try:
        config = json.loads(text)  # JSON is a YAML subset; bootstrap needs no dependency.
    except json.JSONDecodeError:
        import yaml
        config = yaml.safe_load(text)
    parent = config.pop("extends", None)
    if parent:
        config = merge(load_config(path.parent / parent, _seen | {path}), config)
    validate_config(config)
    return config


def validate_config(c: dict) -> None:
    if "timing" not in c:
        return
    t = c["timing"]
    for key in ("sta_sample_s", "history_s", "ap_sample_s", "decision_s", "report_min_s"):
        if t[key] <= 0:
            raise ValueError(f"timing.{key} must be positive")
    ratio = t["history_s"] / t["sta_sample_s"]
    if abs(ratio - round(ratio)) > 1e-8:
        raise ValueError("History must contain an integer number of samples")
    if c["network"]["num_sta"] < 1:
        raise ValueError("At least one STA is required")
    state=c["state"]
    if len(state.get("global_features",[]))!=10 or state.get("associated_sta_scale",0)<=0 or state.get("queue_byte_scale",0)<=0:
        raise ValueError("State schema requires ten AP-local global features and positive scales")
    training=c["training"]
    if training.get("target_update_mode") not in ("hard","soft"):
        raise ValueError("target_update_mode must be hard or soft")
    if training["target_update_steps"]<1 or not 0<training["target_tau"]<=1:
        raise ValueError("Target update interval and tau must be valid")
    reward=c["reward"]; version=reward.get("version","v1")
    if version not in ("v1","v2","v2_service","tgax_v1","tgax_v2_service_guard","tgax_reward_v3",
                       "tgax_cmdp_v1", "tgax_cmdp_v2", "tgax_cmdp_directional_v3",
                       "tgax_cmdp_two_cost_v4", "tgax_cmdp_three_cost_v5", "tgax_cmdp_five_cost_v6"):
        raise ValueError("unsupported reward.version")
    if version in ("tgax_cmdp_two_cost_v4", "tgax_cmdp_three_cost_v5", "tgax_cmdp_five_cost_v6"):
        weights=reward.get("utility_weights",[])
        references=reward.get("qos_delay_reference_ms",[])
        budgets=reward.get("budgets",[])
        expected_order = (["qos", "starvation"] if version == "tgax_cmdp_two_cost_v4" else
                          ["qos_avg", "qos_max", "starvation"] if version == "tgax_cmdp_three_cost_v5" else
                          ["qos_vo_dl", "qos_vo_ul", "qos_vi_dl", "qos_vi_ul", "starvation"])
        if reward.get("cost_order") != expected_order:
            raise ValueError(f"{version} cost order is invalid")
        if len(weights)!=4 or any(value<0 for value in weights) or abs(sum(weights)-1.0)>1e-12:
            raise ValueError("Two-Cost CMDP utility weights must sum to one")
        if len(references)!=2 or any(value<=0 for value in references):
            raise ValueError("Two-Cost CMDP QoS references must contain positive VO/VI values")
        if len(budgets)!=len(expected_order) or any(not 0<=value<=1 for value in budgets):
            raise ValueError(f"{version} budgets must contain {len(expected_order)} values in [0,1]")
        required=()
    elif version in ("tgax_cmdp_v2", "tgax_cmdp_directional_v3"):
        weights=reward.get("utility_weights",[])
        references=reward.get("delay_reference_ms",[])
        minimums=reward.get("minimum_satisfaction",[])
        budgets=reward.get("budgets",[])
        expected_cost_order = (["delay", "loss", "fairness", "starvation"]
            if version == "tgax_cmdp_v2" else
            ["delay_ul", "delay_dl", "loss_ul", "loss_dl", "fairness", "starvation"])
        if reward.get("cost_order") != expected_cost_order:
            raise ValueError(f"{version} cost order is invalid")
        if len(weights)!=4 or any(value<0 for value in weights) or abs(sum(weights)-1.0)>1e-12:
            raise ValueError("TGax CMDP-v2 utility weights must be four non-negative values summing to one")
        if len(references)!=2 or any(value<=0 for value in references) or reward.get("delay_excess_cap",0)<=0:
            raise ValueError("TGax CMDP-v2 delay references and excess cap must be positive")
        if not 0 < reward.get("minimum_jain",0) <= 1:
            raise ValueError("TGax CMDP-v2 minimum Jain must lie in (0,1]")
        if len(minimums)!=2 or any(not 0<value<=1 for value in minimums):
            raise ValueError("TGax CMDP-v2 BE/BK minimum satisfaction must lie in (0,1]")
        if len(budgets)!=len(expected_cost_order) or any(not 0<=value<=1 for value in budgets):
            raise ValueError(f"{version} budgets must contain {len(expected_cost_order)} values in [0,1]")
        required=()
    elif version == "tgax_cmdp_v1":
        weights=reward.get("utility_weights",[]); low=reward.get("low_priority_min_satisfaction",[])
        if len(weights)!=4 or any(value<0 for value in weights) or abs(sum(weights)-1.0)>1e-12:
            raise ValueError("TGax CMDP utility weights must be four non-negative values summing to one")
        if len(low)!=2 or any(not 0<=value<1 for value in low):
            raise ValueError("TGax CMDP BE/BK minimum satisfaction must lie in [0,1)")
        budgets=reward.get("budgets")
        if budgets is not None and (len(budgets)!=4 or any(value<0 for value in budgets)):
            raise ValueError("TGax CMDP budgets must be null before calibration or four non-negative values")
        required=()
    elif version == "tgax_reward_v3":
        required=("on_time_vo","on_time_vi","service_all","jain")
    elif version in ("tgax_v1","tgax_v2_service_guard"):
        required=("service","on_time_vo","on_time_vi","retransmission","final_failure",
                  "low_priority","low_priority_service_floor","action_change","eps")
    else:
        required=("throughput","collision","drop","unfairness","action_change","eps")
        if version!="v1": required+=("on_time_vo","on_time_vi","low_priority","low_priority_service_floor")
    if any(reward.get(key,-1)<0 for key in required):
        raise ValueError("Reward weights, epsilon and service floor must be non-negative")
    if version == "tgax_reward_v3" and abs(sum(reward[key] for key in required)-1.0)>1e-12:
        raise ValueError("TGax Reward-v3 weights must sum to one")
    if version in ("v2_service","tgax_v2_service_guard") and (reward["low_priority"]<=0 or not 0<reward["low_priority_service_floor"]<=1):
        raise ValueError(f"{version} requires a positive low-priority weight and a service floor in (0,1]")
    if version in ("tgax_cmdp_v1", "tgax_cmdp_v2", "tgax_cmdp_directional_v3",
                   "tgax_cmdp_two_cost_v4", "tgax_cmdp_three_cost_v5") and reward.get("budgets") is not None:
        initial=reward.get("multiplier_initial",[])
        if len(initial)!=len(reward["budgets"]) or any(not math.isfinite(value) or value<0 for value in initial):
            raise ValueError("TGax CMDP requires one finite non-negative multiplier per cost")
        if reward.get("multiplier_learning_rate",0)<=0:
            raise ValueError("TGax CMDP multiplier learning rate must be positive")
        if not 0<reward.get("multiplier_ema_alpha",0)<=1:
            raise ValueError("TGax CMDP multiplier EMA alpha must lie in (0,1]")
        if reward.get("multiplier_maximum",0)<=0:
            raise ValueError("TGax CMDP multiplier maximum must be positive")


def digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def backend_source_hash() -> str:
    paths=("ns3_ext/wifi-control.cc", "ns3_ext/cw_growth.patch", "ns3_ext/queue_drop_trace.patch",
           "ns3_ext/internal_collision_trace.patch")
    return hashlib.sha256(b"".join((ROOT/p).read_bytes() for p in paths)).hexdigest()
