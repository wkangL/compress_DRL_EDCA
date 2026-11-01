"""Constrained reward for the direct-MAC TGax-arrival environment.

Replay stores raw utility and costs; current Lagrange multipliers are applied
when a batch is sampled, so historical samples do not retain stale weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


COST_ORDER = ("delay", "loss", "fairness", "starvation")
AC_NAMES = ("vo", "vi", "be", "bk")


def _finite_unit(values, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain four finite values")
    if ((result < 0) | (result > 1)).any():
        raise ValueError(f"{name} must lie in [0,1]")
    return result


def _utility(metrics: dict, config: dict) -> tuple[float, np.ndarray, np.ndarray]:
    satisfaction = _finite_unit(metrics["demand_satisfaction"], "demand_satisfaction")
    active = np.asarray(metrics["demand_active"], dtype=bool)
    if active.shape != (4,):
        raise ValueError("demand_active must contain four flags")
    weights = np.asarray(config["utility_weights"], dtype=np.float64)
    if weights.shape != (4,) or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("utility_weights must be four non-negative values")
    active_weights = weights * active
    concave = np.log1p(9.0 * satisfaction) / np.log(10.0)
    utility = float(np.dot(active_weights, concave) / active_weights.sum()) if active_weights.sum() else 0.0
    return utility, satisfaction, active


def cmdp_components(metrics: dict, config: dict) -> tuple[float, np.ndarray, dict]:
    """Return CMDP-v2 utility and costs ordered delay/loss/fairness/starvation.

    Delay is the equal-weight mean of VO/VI UL/DL group means.  Successful
    MPDUs use normalized service-delay excess.  An active group with no ACKed
    MPDU receives the maximum group delay pressure instead of a false zero.
    Loss is the equal-weight final-failure fraction among MPDUs resolved in
    the window; pending MPDUs are neither successes nor failures.
    """
    utility, satisfaction, active = _utility(metrics, config)
    references = np.asarray(config["delay_reference_ms"], dtype=np.float64)
    cap = float(config["delay_excess_cap"])
    if references.shape != (2,) or (references <= 0).any() or not np.isfinite(references).all():
        raise ValueError("delay_reference_ms must contain positive VO/VI references")
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("delay_excess_cap must be positive")

    delay_groups: list[float] = []
    loss_groups: list[float] = []
    parts: dict[str, object] = {
        "utility": utility,
        **{f"satisfaction_{ac}": float(satisfaction[i]) for i, ac in enumerate(AC_NAMES)},
    }
    for ac_index, ac in enumerate(("vo", "vi")):
        for direction in ("ul", "dl"):
            prefix = f"{direction}_{ac}"
            samples = np.asarray(metrics.get(f"{prefix}_delay_samples_ms", []), dtype=np.float64)
            if samples.ndim != 1 or not np.isfinite(samples).all() or (samples < 0).any():
                raise ValueError(f"{prefix} delay samples must be finite non-negative values")
            risk = int(metrics.get(f"{prefix}_pending_start", 0)) + int(metrics.get(f"{prefix}_arrived", 0))
            if samples.size:
                group_delay = float(np.mean(np.minimum(
                    np.maximum(samples / references[ac_index] - 1.0, 0.0), cap) / cap))
                missing_guard = False
                delay_groups.append(group_delay)
            elif risk > 0:
                # No-success safeguard: an active direction cannot look like
                # zero delay merely because it delivered nothing.
                group_delay = 1.0
                missing_guard = True
                delay_groups.append(group_delay)
            else:
                group_delay = None
                missing_guard = False
            acked = int(metrics.get(f"{prefix}_acked", 0))
            failed = int(metrics.get(f"{prefix}_failed", 0))
            if min(acked, failed, risk) < 0 or acked + failed > risk:
                raise ValueError(f"Invalid {prefix} final-outcome ledger")
            finalized = acked + failed
            group_loss = failed / finalized if finalized else None
            if finalized:
                loss_groups.append(float(group_loss))
            group = f"{ac}_{direction}"
            parts.update({f"delay_{group}": group_delay,
                          f"delay_{group}_valid": bool(risk > 0),
                          f"delay_{group}_missing_success_guard": missing_guard,
                          f"loss_{group}": group_loss,
                          f"loss_{group}_valid": bool(finalized),
                          f"loss_{group}_finalized": finalized,
                          f"loss_{group}_pending_end": int(metrics.get(f"{prefix}_pending_end", 0))})

    delay_valid = bool(delay_groups)
    loss_valid = bool(loss_groups)
    delay_cost = float(np.mean(delay_groups)) if delay_valid else 0.0
    loss_cost = float(np.mean(loss_groups)) if loss_valid else 0.0

    minimum_jain = float(config["minimum_jain"])
    if not np.isfinite(minimum_jain) or not 0 < minimum_jain <= 1:
        raise ValueError("minimum_jain must lie in (0,1]")
    jain = float(np.clip(metrics.get("jain_demand_satisfaction", 0.0), 0.0, 1.0))
    fairness_valid = int(metrics.get("jain_active_sta_count", 0)) > 0
    fairness_cost = max(minimum_jain - jain, 0.0) / minimum_jain if fairness_valid else 0.0

    low_targets = np.asarray(config["minimum_satisfaction"], dtype=np.float64)
    if low_targets.shape != (2,) or ((low_targets <= 0) | (low_targets > 1)).any():
        raise ValueError("minimum_satisfaction must contain BE/BK targets in (0,1]")
    starvation_groups = []
    for offset, ac in enumerate(("be", "bk")):
        index = offset + 2
        valid = bool(active[index])
        deficit = (max(low_targets[offset] - satisfaction[index], 0.0) / low_targets[offset]
                   if valid else None)
        if valid:
            starvation_groups.append(float(deficit))
        parts.update({f"starvation_deficit_{ac}": deficit,
                      f"starvation_{ac}_valid": valid})
    starvation_valid = bool(starvation_groups)
    starvation_cost = float(np.mean(starvation_groups)) if starvation_valid else 0.0

    costs = np.asarray((delay_cost, loss_cost, fairness_cost, starvation_cost), dtype=np.float64)
    valid = np.asarray((delay_valid, loss_valid, fairness_valid, starvation_valid), dtype=bool)
    if not np.isfinite(costs).all() or ((costs < 0) | (costs > 1)).any():
        raise FloatingPointError("CMDP-v2 costs must be finite values in [0,1]")
    parts.update(cost_order=list(COST_ORDER), cost_valid=valid.tolist(), jain=jain,
                 cost_delay=delay_cost, cost_loss=loss_cost,
                 cost_fairness=fairness_cost, cost_starvation=starvation_cost)
    return utility, costs, parts


def legacy_cmdp_components_v1(metrics: dict, config: dict) -> tuple[float, np.ndarray, dict]:
    """Historical timely-delivery CMDP retained only for old checkpoints."""
    utility, satisfaction, _ = _utility(metrics, config)

    valid_vo = bool(metrics.get("vo_on_time_delivery_valid", False))
    valid_vi = bool(metrics.get("vi_on_time_delivery_valid", False))
    timely_vo = float(np.clip(metrics.get("vo_on_time_delivery_rate", 0.0), 0.0, 1.0))
    timely_vi = float(np.clip(metrics.get("vi_on_time_delivery_rate", 0.0), 0.0, 1.0))
    jain = float(np.clip(metrics.get("jain_demand_satisfaction", 0.0), 0.0, 1.0))
    low_targets = np.asarray(config["low_priority_min_satisfaction"], dtype=np.float64)
    if low_targets.shape != (2,) or ((low_targets < 0) | (low_targets >= 1)).any():
        raise ValueError("low_priority_min_satisfaction must contain BE/BK targets in [0,1)")
    low_cost = max((1.0 - satisfaction[2]) / (1.0 - low_targets[0]),
                   (1.0 - satisfaction[3]) / (1.0 - low_targets[1]))
    if bool(metrics.get("be_or_bk_zero_service_streak_violation", False)):
        low_cost = max(low_cost, 2.0)
    costs = np.asarray((1.0 - timely_vo, 1.0 - timely_vi, 1.0 - jain, low_cost), dtype=np.float64)
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise FloatingPointError("Non-finite CMDP cost")
    parts = {
        "utility": utility,
        **{f"satisfaction_{ac}": float(satisfaction[i]) for i, ac in enumerate(AC_NAMES)},
        "timely_vo": timely_vo, "timely_vi": timely_vi,
        "timely_vo_valid": valid_vo, "timely_vi_valid": valid_vi, "jain": jain,
        "cost_vo": float(costs[0]), "cost_vi": float(costs[1]),
        "cost_jain": float(costs[2]), "cost_low_priority": float(costs[3]),
    }
    return utility, costs, parts


def lagrangian_reward(utility: float, costs, budgets, multipliers,
                      *, excess_only: bool = False) -> float:
    """Combine raw CMDP components using the current multipliers.

    CMDP-v2 uses ``utility - lambda dot cost``.  Budgets drive the projected
    multiplier update, not the sampled TD reward.  ``excess_only`` exists only
    to reproduce the historical v1 checkpoint semantics.
    """
    costs = np.asarray(costs, dtype=np.float64)
    budgets = np.asarray(budgets, dtype=np.float64)
    multipliers = np.asarray(multipliers, dtype=np.float64)
    if costs.shape != (4,) or budgets.shape != (4,) or multipliers.shape != (4,):
        raise ValueError("CMDP costs, budgets and multipliers must have four entries")
    if not np.isfinite(costs).all() or not np.isfinite(budgets).all() or not np.isfinite(multipliers).all():
        raise FloatingPointError("Non-finite Lagrangian component")
    if (multipliers < 0).any():
        raise ValueError("Lagrange multipliers must be non-negative")
    penalized = np.maximum(0.0, costs - budgets) if excess_only else costs
    return float(utility - np.dot(multipliers, penalized))


@dataclass
class LagrangeController:
    budgets: np.ndarray
    multipliers: np.ndarray
    learning_rate: float = 0.01
    ema_alpha: float = 0.2
    maximum: float = 1.0
    ema_costs: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.budgets = np.asarray(self.budgets, dtype=np.float64)
        self.multipliers = np.asarray(self.multipliers, dtype=np.float64)
        if self.budgets.shape != (4,) or self.multipliers.shape != (4,):
            raise ValueError("Four CMDP budgets and multipliers are required")
        if (not np.isfinite(self.budgets).all() or not np.isfinite(self.multipliers).all()
                or (self.budgets < 0).any() or (self.multipliers < 0).any()):
            raise ValueError("CMDP budgets and multipliers must be finite and non-negative")
        if not 0 < self.ema_alpha <= 1 or self.learning_rate <= 0 or self.maximum <= 0:
            raise ValueError("Invalid multiplier update settings")

    def update(self, episode_costs, valid_mask=None) -> np.ndarray:
        costs = np.asarray(episode_costs, dtype=np.float64)
        mask = np.ones(4, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
        if costs.shape != (4,) or mask.shape != (4,) or not np.isfinite(costs[mask]).all():
            raise ValueError("Invalid episode costs or mask")
        if self.ema_costs is None:
            self.ema_costs = costs.copy()
        else:
            self.ema_costs[mask] = ((1.0 - self.ema_alpha) * self.ema_costs[mask]
                                    + self.ema_alpha * costs[mask])
        self.multipliers[mask] = np.clip(
            self.multipliers[mask] + self.learning_rate * (self.ema_costs[mask] - self.budgets[mask]),
            0.0, self.maximum)
        return self.multipliers.copy()
