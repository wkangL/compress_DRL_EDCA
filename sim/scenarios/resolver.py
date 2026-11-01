"""Resolve legacy capacity-relative and explicit per-STA traffic definitions."""
from __future__ import annotations

import math


def resolve_scenario(definition: dict, num_sta: int, capacity_mbps: float) -> dict:
    """Return the backend traffic fields while retaining descriptive metadata.

    ``per_sta_mbps_by_ac`` is the aggregate UL+DL rate of one station.  Legacy
    ``rho`` definitions remain supported so preserved experiments can replay.
    """
    if num_sta < 1 or not math.isfinite(capacity_mbps) or capacity_mbps <= 0:
        raise ValueError("num_sta and capacity_mbps must be positive")
    result = dict(definition)
    if "per_sta_mbps_by_ac" in definition:
        rates = list(definition["per_sta_mbps_by_ac"])
        if len(rates) != 4 or any(not math.isfinite(float(x)) or float(x) < 0 for x in rates):
            raise ValueError("per_sta_mbps_by_ac must contain four finite nonnegative rates")
        total_per_sta = sum(float(x) for x in rates)
        if total_per_sta <= 0:
            raise ValueError("per_sta_mbps_by_ac must have positive total load")
        result["per_sta_mbps_by_ac"] = rates
        result["offered_mbps"] = total_per_sta * num_sta
        result["ac_mix"] = [float(x) / total_per_sta for x in rates]
        result["load_model"] = "absolute_per_sta"
    elif "rho" in definition and "ac_mix" in definition:
        rho = float(definition["rho"])
        mix = list(definition["ac_mix"])
        if not math.isfinite(rho) or rho <= 0 or len(mix) != 4:
            raise ValueError("legacy rho/ac_mix traffic definition is invalid")
        if any(not math.isfinite(float(x)) or float(x) < 0 for x in mix):
            raise ValueError("ac_mix must contain four finite nonnegative values")
        mix_total = sum(float(x) for x in mix)
        if not math.isclose(mix_total, 1.0, rel_tol=0, abs_tol=1e-8):
            raise ValueError("ac_mix must sum to one")
        result["offered_mbps"] = rho * capacity_mbps
        result["ac_mix"] = [float(x) for x in mix]
        result["load_model"] = "legacy_capacity_relative"
    else:
        raise ValueError("scenario requires per_sta_mbps_by_ac or legacy rho/ac_mix")
    ul = float(result.get("ul_fraction", -1))
    if not math.isfinite(ul) or not 0 <= ul <= 1:
        raise ValueError("ul_fraction must be within [0, 1]")
    pattern=result.get("load_pattern",{"type":"stationary"})
    if pattern.get("type","stationary") not in ("stationary","two_state"):
        raise ValueError("load_pattern.type must be stationary or two_state")
    if pattern.get("type") == "two_state":
        period=float(pattern.get("period_s",0)); low=float(pattern.get("low_scale",0)); high=float(pattern.get("high_scale",0))
        if not all(math.isfinite(x) for x in (period,low,high)) or period<=0 or low<=0 or high<=0 or low>=high:
            raise ValueError("two_state load pattern requires period>0 and 0<low_scale<high_scale")
    return result
