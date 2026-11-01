import pytest

from sim.scenarios.resolver import resolve_scenario


def test_resolve_absolute_per_sta_rates():
    scenario = resolve_scenario(
        {"per_sta_mbps_by_ac": [0.5, 1.0, 0.4, 0.1], "ul_fraction": 0.5}, 10, 27.0)
    assert scenario["offered_mbps"] == pytest.approx(20.0)
    assert scenario["ac_mix"] == pytest.approx([0.25, 0.5, 0.2, 0.05])
    assert scenario["load_model"] == "absolute_per_sta"


def test_resolve_legacy_rho():
    scenario = resolve_scenario(
        {"rho": 0.5, "ac_mix": [0.1, 0.2, 0.6, 0.1], "ul_fraction": 0.6}, 10, 28.0)
    assert scenario["offered_mbps"] == pytest.approx(14.0)
    assert scenario["load_model"] == "legacy_capacity_relative"


@pytest.mark.parametrize("rates", ([1, 2, 3], [0, 0, 0, 0], [1, -1, 1, 1]))
def test_reject_invalid_absolute_rates(rates):
    with pytest.raises(ValueError):
        resolve_scenario({"per_sta_mbps_by_ac": rates, "ul_fraction": 0.5}, 10, 28.0)


def test_two_state_load_pattern_is_preserved():
    pattern={"type":"two_state","period_s":5.0,"low_scale":0.5,"high_scale":1.5,"stagger_by_sta":True}
    scenario=resolve_scenario({"per_sta_mbps_by_ac":[.1,.2,.3,.1],"ul_fraction":.6,"load_pattern":pattern},10,28.0)
    assert scenario["load_pattern"] == pattern


def test_reject_invalid_two_state_pattern():
    with pytest.raises(ValueError):
        resolve_scenario({"per_sta_mbps_by_ac":[.1,.2,.3,.1],"ul_fraction":.6,
                          "load_pattern":{"type":"two_state","period_s":0,"low_scale":1,"high_scale":.5}},10,28.0)
