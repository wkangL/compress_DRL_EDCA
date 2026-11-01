from __future__ import annotations
import numpy as np


def demand_satisfaction(metrics: dict, config: dict) -> tuple[float,np.ndarray,np.ndarray]:
    acs=("vo","vi","be","bk")
    offered=np.array([metrics[f"{ac}_offered_mbps"] for ac in acs])
    goodput=np.array([metrics[f"{ac}_goodput_mbps"] for ac in acs])
    active=offered>0; w=np.asarray(config["ac_weights"])*active
    u=np.clip(goodput/(offered+config["eps"]),0,1)
    throughput=float(np.dot(w,u)/w.sum()) if w.sum()>0 else 0.
    return throughput,u,active


def reward_v1(metrics: dict, action_change: float, config: dict) -> tuple[float,dict]:
    """Original duration-normalized reward, retained bit-for-bit as Reward-v1."""
    acs=("vo","vi","be","bk")
    throughput,_,_=demand_satisfaction(metrics,config)
    penalty=[]
    for ac,threshold in zip(acs[:2],config["qos_threshold_ms"]):
        penalty.append(float(np.clip(max(0,metrics[f"{ac}_delay_p95_ms"]/threshold-1),0,2))
                       if metrics[f"{ac}_delay_valid"] else 0.)
    parts=dict(throughput=throughput,collision=float(np.clip(metrics["collision_rate"],0,1)),
               drop=float(np.clip(metrics["drop_rate"],0,1)),vo_delay=penalty[0],vi_delay=penalty[1],
               unfairness=1-metrics["jain_fairness"],action_change=action_change)
    total=config["throughput"]*parts["throughput"]-sum(config[k]*parts[k] for k in parts if k!="throughput")
    return float(total),parts


def reward_v2(metrics: dict, action_change: float, config: dict) -> tuple[float,dict]:
    """Causal deadline reward; every term is known at the current STEP boundary."""
    throughput,satisfaction,active=demand_satisfaction(metrics,config)
    on_time=[]
    for ac in ("vo","vi"):
        valid=bool(metrics.get(f"{ac}_on_time_delivery_valid",False))
        on_time.append(float(np.clip(metrics.get(f"{ac}_on_time_delivery_rate",0.),0,1)) if valid else 0.)
    low_active=active[2:]
    floor=float(config.get("low_priority_service_floor",0.))
    deficit=float(np.mean(np.maximum(0.,floor-satisfaction[2:][low_active]))) if low_active.any() else 0.
    parts=dict(throughput=throughput,on_time_vo=on_time[0],on_time_vi=on_time[1],
               collision=float(np.clip(metrics["collision_rate"],0,1)),
               drop=float(np.clip(metrics.get("final_failure_fraction",metrics["drop_rate"]),0,1)),
               unfairness=float(np.clip(1-metrics["jain_fairness"],0,1)),
               action_change=float(action_change),low_priority_deficit=deficit)
    total=(config["throughput"]*parts["throughput"]
           +config["on_time_vo"]*parts["on_time_vo"]
           +config["on_time_vi"]*parts["on_time_vi"]
           -config["collision"]*parts["collision"]
           -config["drop"]*parts["drop"]
           -config["unfairness"]*parts["unfairness"]
           -config["action_change"]*parts["action_change"]
           -config.get("low_priority",0.)*parts["low_priority_deficit"])
    return float(total),parts


def reward(metrics: dict, action_change: float, config: dict) -> tuple[float,dict]:
    """Dispatch an immutable, explicitly named reward definition."""
    version=config.get("version","v1")
    if version=="v1": return reward_v1(metrics,action_change,config)
    if version in ("v2","v2_service"): return reward_v2(metrics,action_change,config)
    raise ValueError(f"Unsupported reward version: {version}")
