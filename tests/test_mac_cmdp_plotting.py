import math

import pytest

from scripts.analyze_mac_cmdp_experiment import ccdf_lower_percent, relative_improvement


def test_ccdf_floor_follows_successful_sample_count():
    assert ccdf_lower_percent(0) is None
    assert ccdf_lower_percent(19) is None
    assert ccdf_lower_percent(20) == 5.0
    assert ccdf_lower_percent(99) == 5.0
    assert ccdf_lower_percent(100) == 1.0
    assert ccdf_lower_percent(999) == 1.0
    assert ccdf_lower_percent(1000) == 0.1


def test_relative_improvement_uses_metric_direction():
    assert relative_improvement(10.0, 12.0, True) == pytest.approx(20.0)
    assert relative_improvement(10.0, 8.0, False) == pytest.approx(20.0)
    assert math.isnan(relative_improvement(0.0, 1.0, True))
