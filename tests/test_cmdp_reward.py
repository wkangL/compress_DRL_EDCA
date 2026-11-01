from __future__ import annotations

import math
import numpy as np

from rl.cmdp_reward import (COST_ORDER, DIRECTIONAL_COST_ORDER, LagrangeController,
                            cmdp_components, completed_qos_increment, directional_cmdp_components, lagrangian_reward,
                            qos_mpdu_cost, two_cost_cmdp_components)
from rl.replay_buffer import Transition


CONFIG = {
    "utility_weights": [0.4, 0.4, 0.1, 0.1],
    "delay_reference_ms": [20.0, 50.0],
    "delay_excess_cap": 3.0,
    "minimum_jain": 0.5,
    "minimum_satisfaction": [0.02, 0.005],
}


def sample(**updates):
    result = {
        "demand_satisfaction": [0.7, 0.5, 0.02, 0.005],
        "demand_active": [True, True, True, True],
        "jain_demand_satisfaction": 0.6,
        "jain_active_sta_count": 40,
    }
    for ac in ("vo", "vi"):
        for direction in ("ul", "dl"):
            prefix = f"{direction}_{ac}"
            result.update({f"{prefix}_pending_start": 0, f"{prefix}_arrived": 10,
                           f"{prefix}_acked": 10, f"{prefix}_failed": 0,
                           f"{prefix}_pending_end": 0,
                           f"{prefix}_delay_samples_ms": [10.0] * 10})
    result.update(updates)
    return result


def test_priority_weighted_concave_utility_exact_and_cost_order():
    utility, costs, parts = cmdp_components(sample(), CONFIG)
    sat = np.asarray([0.7, 0.5, 0.02, 0.005])
    expected = float(np.dot([0.4, 0.4, 0.1, 0.1], np.log1p(9 * sat) / np.log(10)))
    assert math.isclose(utility, expected, rel_tol=0, abs_tol=1e-12)
    assert parts["cost_order"] == list(COST_ORDER)
    assert costs.shape == (4,) and np.isfinite(costs).all()
    assert ((0 <= costs) & (costs <= 1)).all()


def test_delay_cost_equal_weights_directions_not_packet_counts():
    metrics = sample(ul_vo_delay_samples_ms=[10.0] * 100,
                     dl_vo_delay_samples_ms=[80.0], dl_vo_arrived=1, dl_vo_acked=1)
    _, costs, parts = cmdp_components(metrics, CONFIG)
    assert math.isclose(parts["delay_vo_dl"], 1.0)
    assert math.isclose(costs[0], 0.25)


def test_active_direction_without_success_gets_maximum_delay_pressure():
    metrics = sample(ul_vi_delay_samples_ms=[], ul_vi_acked=0,
                     ul_vi_failed=0, ul_vi_pending_end=10)
    _, costs, parts = cmdp_components(metrics, CONFIG)
    assert parts["delay_vi_ul"] == 1.0
    assert parts["delay_vi_ul_missing_success_guard"] is True
    assert parts["loss_vi_ul"] is None
    assert parts["cost_valid"] == [True, True, True, True]
    assert costs[0] == 0.25


def test_loss_uses_final_outcomes_only_and_pending_is_not_loss():
    metrics = sample(ul_vo_arrived=12, ul_vo_acked=8, ul_vo_failed=2,
                     ul_vo_pending_end=2, ul_vo_delay_samples_ms=[10.0] * 8)
    _, costs, parts = cmdp_components(metrics, CONFIG)
    assert math.isclose(parts["loss_vo_ul"], 0.2)
    assert math.isclose(costs[1], 0.05)
    assert parts["loss_vo_ul_pending_end"] == 2


def test_resolved_failure_is_not_recounted_in_the_next_pending_window():
    first = sample(ul_vo_pending_start=5, ul_vo_arrived=5, ul_vo_acked=4,
                   ul_vo_failed=1, ul_vo_pending_end=5,
                   ul_vo_delay_samples_ms=[10.0] * 4)
    _, _, first_parts = cmdp_components(first, CONFIG)
    second = sample(ul_vo_pending_start=5, ul_vo_arrived=0, ul_vo_acked=0,
                    ul_vo_failed=0, ul_vo_pending_end=5,
                    ul_vo_delay_samples_ms=[])
    _, _, second_parts = cmdp_components(second, CONFIG)
    assert math.isclose(first_parts["loss_vo_ul"], 0.2)
    assert second_parts["loss_vo_ul"] is None
    assert second_parts["loss_vo_ul_finalized"] == 0


def test_be_and_bk_starvation_are_separate_before_equal_mean():
    _, healthy, _ = cmdp_components(sample(), CONFIG)
    _, bk_bad, parts = cmdp_components(
        sample(demand_satisfaction=[0.7, 0.5, 1.0, 0.0]), CONFIG)
    assert healthy[3] == 0.0
    assert parts["starvation_deficit_be"] == 0.0
    assert parts["starvation_deficit_bk"] == 1.0
    assert math.isclose(bk_bad[3], 0.5)


def test_inactive_low_priority_group_is_excluded_not_given_full_credit():
    _, costs, parts = cmdp_components(
        sample(demand_satisfaction=[0.7, 0.5, 0.0, 0.0],
               demand_active=[True, True, False, True]), CONFIG)
    assert parts["starvation_deficit_be"] is None
    assert parts["starvation_deficit_bk"] == 1.0
    assert costs[3] == 1.0


def test_fairness_cost_is_normalized_deficit():
    _, costs, _ = cmdp_components(sample(jain_demand_satisfaction=0.4), CONFIG)
    assert math.isclose(costs[2], 0.2)
    _, missing, parts = cmdp_components(sample(jain_active_sta_count=0), CONFIG)
    assert missing[2] == 0.0 and parts["cost_valid"][2] is False


def test_lagrangian_td_reward_uses_raw_costs_not_budget_subtracted_costs():
    reward = lagrangian_reward(0.8, [0.4, 0.1, 0.2, 0.5],
                               [0.3, 0.15, 0.0, 0.25], [1, 1, 0.5, 0.25])
    assert math.isclose(reward, 0.8 - 0.4 - 0.1 - 0.5 * 0.2 - 0.25 * 0.5,
                        abs_tol=1e-12)


def test_legacy_excess_only_combination_remains_available():
    reward = lagrangian_reward(0.8, [0.4, 0.1, 0.2, 0.5],
                               [0.3, 0.15, 0.0, 0.25], [1, 1, 0.5, 0.25],
                               excess_only=True)
    assert math.isclose(reward, 0.8 - 0.1 - 0.5 * 0.2 - 0.25 * 0.25,
                        abs_tol=1e-12)


def test_multiplier_moves_up_down_and_respects_valid_mask():
    controller = LagrangeController(np.asarray([0.3, 0.15, 0.0, 0.25]), np.full(4, 0.1),
                                    learning_rate=0.1, ema_alpha=1.0, maximum=1.0)
    first = controller.update([0.5, 0.5, 0.0, 0.8], valid_mask=[True, False, True, True])
    assert first[0] > 0.1 and first[1] == 0.1 and first[2] == 0.1 and first[3] > 0.1
    second = controller.update([0.0, 0.0, 0.0, 0.0])
    assert second[0] < first[0] and second[1] < first[1] and second[3] < first[3]


def test_transition_retains_raw_components_and_current_lambda_recombines():
    costs = np.asarray([0.4, 0.1, 0.2, 0.5])
    transition = Transition({}, np.zeros(5, dtype=np.int64), 0.0, {}, False, 0,
                            utility=0.75, costs=costs)
    first = lagrangian_reward(transition.utility, transition.costs,
                              [0.3, 0.15, 0.0, 0.25], [0.1] * 4)
    second = lagrangian_reward(transition.utility, transition.costs,
                               [0.3, 0.15, 0.0, 0.25], [1.0] * 4)
    assert transition.utility == 0.75 and transition.costs.shape == (4,)
    assert second < first


def test_directional_cost_prevents_bad_ul_group_from_hiding_behind_dl():
    metrics = sample(ul_vo_delay_samples_ms=[80.0] * 10)
    _, aggregate, _ = cmdp_components(metrics, CONFIG)
    _, directional, parts = directional_cmdp_components(metrics, CONFIG)
    assert math.isclose(aggregate[0], 0.25)  # passes an aggregate 0.40 budget
    assert math.isclose(directional[0], 0.50)  # violates the UL 0.40 budget
    assert directional[1] == 0.0
    assert parts["cost_order"] == list(DIRECTIONAL_COST_ORDER)
    assert parts["cost_valid"] == [True] * 6


def test_directional_final_loss_does_not_compensate_across_ul_and_dl():
    metrics = sample(ul_vo_acked=8, ul_vo_failed=2,
                     ul_vo_delay_samples_ms=[10.0] * 8)
    _, aggregate, _ = cmdp_components(metrics, CONFIG)
    _, directional, parts = directional_cmdp_components(metrics, CONFIG)
    assert math.isclose(aggregate[1], 0.05)
    assert math.isclose(directional[2], 0.10)
    assert directional[3] == 0.0
    assert math.isclose(parts["loss_vo_ul"], 0.2)


def test_six_cost_lagrange_controller_and_reward_are_supported():
    budgets = np.asarray([0.4, 0.4, 0.08, 0.08, 0.0, 0.3])
    initial = np.asarray([0.1, 0.1, 0.15, 0.15, 0.1, 0.2])
    costs = np.asarray([0.5, 0.2, 0.1, 0.0, 0.2, 0.6])
    controller = LagrangeController(budgets, initial, learning_rate=0.05,
                                    ema_alpha=1.0, maximum=3.0)
    updated = controller.update(costs)
    assert updated.shape == (6,)
    assert updated[0] > initial[0] and updated[1] < initial[1]
    assert updated[2] > initial[2] and updated[3] < initial[3]
    reward = lagrangian_reward(0.8, costs, budgets, updated)
    assert math.isfinite(reward) and reward < 0.8


def two_cost_sample(**updates):
    result = sample()
    for ac in ("vo", "vi"):
        for direction in ("ul", "dl"):
            prefix = f"qos_{ac}_{direction}"
            result.update({prefix: 0.2, f"{prefix}_valid": True,
                           f"{prefix}_finalized": 10, f"{prefix}_pending_end": 0})
    result.update(updates)
    return result


def test_qos_mpdu_cost_is_monotone_and_drop_endpoint_is_one():
    assert qos_mpdu_cost(10.0, 20.0) < qos_mpdu_cost(40.0, 20.0)
    assert qos_mpdu_cost(200.0, 20.0) > qos_mpdu_cost(80.0, 20.0)
    assert math.isclose(qos_mpdu_cost(10.0, 20.0), 1.0 / 3.0)
    # A final drop is added as unit cost by the exact completion ledger.
    success_sum = qos_mpdu_cost(10.0, 20.0)
    assert math.isclose((success_sum + 1.0) / 2.0, 2.0 / 3.0)


def test_two_cost_qos_preserves_four_direction_groups():
    metrics = two_cost_sample(qos_vo_ul=0.1, qos_vo_dl=0.9,
                              qos_vi_ul=0.2, qos_vi_dl=0.4)
    _, costs, parts = two_cost_cmdp_components(metrics, CONFIG)
    assert math.isclose(costs[0], 0.4)
    assert parts["qos_vo_ul"] == 0.1 and parts["qos_vo_dl"] == 0.9
    assert parts["cost_order"] == ["qos", "starvation"]


def test_two_cost_pending_only_group_is_invalid_not_false_zero():
    metrics = two_cost_sample(qos_vo_dl=None, qos_vo_dl_valid=False,
                              qos_vo_dl_finalized=0, qos_vo_dl_pending_end=8)
    _, costs, parts = two_cost_cmdp_components(metrics, CONFIG)
    assert math.isclose(costs[0], 0.2)  # mean of the three valid groups
    assert parts["qos_vo_dl"] is None
    assert parts["qos_vo_dl_pending_end"] == 8


def test_pending_mpdu_is_not_counted_or_duplicated_before_final_outcome():
    total_sum = 0.0; total_count = 0
    increment,count = completed_qos_increment([], 0, 20.0)  # pending-only window
    total_sum += increment; total_count += count
    assert total_sum == 0.0 and total_count == 0
    increment,count = completed_qos_increment([], 1, 20.0)  # later final drop
    total_sum += increment; total_count += count
    assert total_sum == 1.0 and total_count == 1
    increment,count = completed_qos_increment([], 0, 20.0)  # subsequent window
    total_sum += increment; total_count += count
    assert total_sum == 1.0 and total_count == 1


def test_two_cost_starvation_uses_worst_active_low_priority_ac():
    _, costs, _ = two_cost_cmdp_components(
        two_cost_sample(demand_satisfaction=[0.7, 0.5, 0.8, 0.0]), CONFIG)
    assert costs[1] == 1.0
    _, inactive, parts = two_cost_cmdp_components(
        two_cost_sample(demand_satisfaction=[0.7, 0.5, 0.0, 0.0],
                        demand_active=[True, True, False, False]), CONFIG)
    assert inactive[1] == 0.0 and parts["cost_valid"][1] is False


def test_two_cost_multiplier_and_replay_recombine_current_lambda():
    controller = LagrangeController(np.asarray([0.4, 0.9]), np.asarray([0.1, 0.1]),
                                    learning_rate=0.1, ema_alpha=1.0, maximum=3.0)
    up = controller.update([0.6, 1.0])
    assert up[0] > 0.1 and up[1] > 0.1
    down = controller.update([0.1, 0.2])
    assert down[0] < up[0] and down[1] < up[1]
    transition = Transition({}, np.zeros(5, dtype=np.int64), 99.0, {}, False, 0,
                            utility=0.8, costs=np.asarray([0.5, 0.7]))
    first = lagrangian_reward(transition.utility, transition.costs,
                              [0.4, 0.9], [0.1, 0.1])
    second = lagrangian_reward(transition.utility, transition.costs,
                               [0.4, 0.9], [1.0, 1.0])
    assert second < first < transition.utility
