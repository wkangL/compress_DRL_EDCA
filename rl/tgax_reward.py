"""Auditable TGax BSS-0 rewards, selected explicitly by config version."""
from __future__ import annotations
import numpy as np


def tgax_reward(metrics: dict, action_change: float, config: dict) -> tuple[float,dict]:
    if config.get("version") == "tgax_reward_v3":
        satisfaction=np.asarray(metrics["application_satisfaction"],dtype=float)
        active=np.asarray(metrics["application_demand_active"],dtype=bool)
        # An AC with no available exogenous demand is omitted, never assigned 1.
        service=float(satisfaction[active].mean()) if active.any() else 0.
        timely_vo=(float(metrics["vo_on_time_delivery_rate"])
                   if metrics.get("vo_on_time_delivery_valid",False) else 0.)
        timely_vi=(float(metrics["vi_on_time_delivery_rate"])
                   if metrics.get("vi_on_time_delivery_valid",False) else 0.)
        jain=float(np.clip(metrics.get("jain_demand_satisfaction",0.),0.,1.))
        parts={"on_time_vo":float(np.clip(timely_vo,0.,1.)),
               "on_time_vi":float(np.clip(timely_vi,0.,1.)),
               "service_all":float(np.clip(service,0.,1.)),
               "jain":jain}
        total=sum(float(config[key])*parts[key] for key in parts)
        if not np.isfinite(total): raise FloatingPointError("Nonfinite TGax Reward-v3")
        if not 0. <= total <= 1.+1e-12: raise ValueError(f"TGax Reward-v3 outside [0,1]: {total}")
        return float(np.clip(total,0.,1.)),parts

    satisfaction=np.asarray(metrics["application_satisfaction"],dtype=float)
    active=np.asarray(metrics["application_demand_active"],dtype=bool)
    weights=np.asarray(config["ac_weights"],dtype=float)*active
    service=float(np.dot(weights,satisfaction)/weights.sum()) if weights.sum()>0 else 0.
    low=active[2:]
    floor=float(config["low_priority_service_floor"])
    low_deficit=float(np.mean(np.maximum(0.,floor-satisfaction[2:][low]))) if low.any() else 0.
    raw_vo=float(metrics["vo_on_time_delivery_rate"])
    raw_vi=float(metrics["vi_on_time_delivery_rate"])
    if config.get("couple_timely_to_application_service",False):
        # A controller must not improve its deadline score by keeping demand
        # above the MAC and thereby shrinking the population of successful MPDUs.
        timely_vo=raw_vo*float(satisfaction[0]) if active[0] else 0.
        timely_vi=raw_vi*float(satisfaction[1]) if active[1] else 0.
    else:
        timely_vo, timely_vi=raw_vo, raw_vi
    parts=dict(service=service,on_time_vo=timely_vo,on_time_vi=timely_vi,
        raw_on_time_vo=raw_vo,raw_on_time_vi=raw_vi,
        retransmission=float(np.clip(metrics["retransmission_rate"],0,1)),
        final_failure=float(np.clip(metrics["final_failure_fraction"],0,1)),
        low_priority_deficit=low_deficit,action_change=float(np.clip(action_change,0,1)))
    total=(config["service"]*parts["service"]+config["on_time_vo"]*parts["on_time_vo"]+
        config["on_time_vi"]*parts["on_time_vi"]-config["retransmission"]*parts["retransmission"]-
        config["final_failure"]*parts["final_failure"]-
        config["low_priority"]*parts["low_priority_deficit"]-
        config["action_change"]*parts["action_change"])
    if not np.isfinite(total): raise FloatingPointError("Nonfinite TGax reward")
    return float(total),parts
