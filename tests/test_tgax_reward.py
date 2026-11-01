from __future__ import annotations

import math

from rl.tgax_reward import tgax_reward


CONFIG = {
    "ac_weights": [1.2, 1.1, 1.0, 0.5],
    "low_priority_service_floor": 0.2,
    "service": 1.0,
    "on_time_vo": 0.8,
    "on_time_vi": 1.0,
    "retransmission": 0.3,
    "final_failure": 0.3,
    "low_priority": 0.5,
    "action_change": 0.02,
}

V3 = {
    "version": "tgax_reward_v3",
    "on_time_vo": 0.40,
    "on_time_vi": 0.40,
    "service_all": 0.10,
    "jain": 0.10,
}


def metrics(active=(True, True, True, True), satisfaction=(1.0, 1.0, 1.0, 1.0)):
    return {
        "application_satisfaction": list(satisfaction),
        "application_demand_active": list(active),
        "vo_on_time_delivery_rate": 1.0,
        "vi_on_time_delivery_rate": 1.0,
        "retransmission_rate": 0.0,
        "final_failure_fraction": 0.0,
    }


def test_active_unserved_low_priority_is_penalized():
    value, parts = tgax_reward(metrics(satisfaction=(1.0, 1.0, 0.0, 0.0)), 0.0, CONFIG)
    assert math.isfinite(value)
    assert parts["low_priority_deficit"] == 0.2
    assert value < tgax_reward(metrics(), 0.0, CONFIG)[0]


def test_inactive_ac_is_excluded_instead_of_fabricating_zero_service():
    _, parts = tgax_reward(
        metrics(active=(True, True, False, False), satisfaction=(0.5, 0.5, 0.0, 0.0)),
        0.0,
        CONFIG,
    )
    assert parts["service"] == 0.5
    assert parts["low_priority_deficit"] == 0.0


def test_action_change_and_rate_terms_have_auditable_scale():
    clean, _ = tgax_reward(metrics(), 0.0, CONFIG)
    degraded, parts = tgax_reward({**metrics(), "retransmission_rate": 1.0,
                                   "final_failure_fraction": 1.0}, 1.0, CONFIG)
    assert parts["action_change"] == 1.0
    assert math.isclose(clean - degraded, 0.62, rel_tol=0, abs_tol=1e-12)


def test_timely_reward_cannot_hide_missing_application_service():
    guarded = {**CONFIG, "couple_timely_to_application_service": True}
    _, healthy = tgax_reward(metrics(satisfaction=(0.8, 0.7, 1.0, 1.0)), 0.0, guarded)
    _, starved = tgax_reward(metrics(satisfaction=(0.1, 0.05, 1.0, 1.0)), 0.0, guarded)
    assert healthy["raw_on_time_vo"] == starved["raw_on_time_vo"] == 1.0
    assert healthy["on_time_vo"] == 0.8
    assert starved["on_time_vo"] == 0.1
    assert healthy["on_time_vi"] == 0.7
    assert starved["on_time_vi"] == 0.05


def test_reward_v3_exact_formula_and_range():
    sample = {**metrics(active=(True, True, True, True),
                        satisfaction=(0.8, 0.6, 0.4, 0.2)),
              "vo_on_time_delivery_rate": 0.75,
              "vi_on_time_delivery_rate": 0.5,
              "vo_on_time_delivery_valid": True,
              "vi_on_time_delivery_valid": True,
              "jain_demand_satisfaction": 0.6}
    value, parts = tgax_reward(sample, 1.0, V3)
    assert parts["on_time_vo"] == 0.75
    assert parts["on_time_vi"] == 0.5
    assert math.isclose(parts["service_all"], 0.5, abs_tol=1e-12)
    assert parts["jain"] == 0.6
    assert math.isclose(value, 0.61, rel_tol=0, abs_tol=1e-12)


def test_reward_v3_never_rewards_missing_evidence_as_perfect():
    sample = {**metrics(active=(False, False, False, False),
                        satisfaction=(0.0, 0.0, 0.0, 0.0)),
              "vo_on_time_delivery_valid": False,
              "vi_on_time_delivery_valid": False,
              "jain_demand_satisfaction": 0.0}
    value, parts = tgax_reward(sample, 0.0, V3)
    assert value == 0.0
    assert all(term == 0.0 for term in parts.values())


def test_reward_v3_penalizes_old_conservative_policy_example():
    default = {**metrics(satisfaction=(0.8, 0.7, 0.4, 0.3)),
               "vo_on_time_delivery_rate": 0.841,
               "vi_on_time_delivery_rate": 0.983,
               "vo_on_time_delivery_valid": True,
               "vi_on_time_delivery_valid": True,
               "jain_demand_satisfaction": 0.239}
    old_ddqn = {**metrics(satisfaction=(0.7, 0.6, 0.3, 0.2)),
                "vo_on_time_delivery_rate": 0.275,
                "vi_on_time_delivery_rate": 0.331,
                "vo_on_time_delivery_valid": True,
                "vi_on_time_delivery_valid": True,
                "jain_demand_satisfaction": 0.161}
    assert tgax_reward(old_ddqn, 0.0, V3)[0] < tgax_reward(default, 0.0, V3)[0]
