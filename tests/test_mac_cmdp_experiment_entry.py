from pathlib import Path

import yaml

from project_config import load_config
from scripts.run_mac_cmdp_experiment import assigned_sta, paths, prepare, validate


ROOT = Path(__file__).resolve().parents[1]


def test_active_yaml_validates_and_covers_all_access_categories():
    spec = yaml.safe_load((ROOT / "configs/mac_cmdp_experiment.yaml").read_text(encoding="utf-8"))
    validate(spec)
    assert sum(int(row["count"]) for row in spec["traffic"]) > 0
    assert {row["ac"] for row in spec["traffic"] if int(row["count"]) > 0} == {"VO", "VI", "BE", "BK"}
    assert spec["training"]["parallel_workers"] == 3


def test_all_derived_paths_live_under_one_output_root(tmp_path):
    spec = yaml.safe_load((ROOT / "configs/mac_cmdp_experiment.yaml").read_text(encoding="utf-8"))
    derived = paths(spec, tmp_path / "experiment")
    assert derived["root"] == (tmp_path / "experiment").resolve()
    assert all(path == derived["root"] or derived["root"] in path.parents for path in derived.values())


def test_assignment_modes_are_deterministic_and_bounded():
    spread = {"assignment": "spread", "sta_offset": 2}
    assert [assigned_sta(spread, i, 10) for i in range(10)] == [2, 6, 10, 14, 18, 22, 26, 30, 34, 38]
    round_robin = {"assignment": "round_robin", "sta_offset": 39}
    assert [assigned_sta(round_robin, i, 3) for i in range(3)] == [39, 0, 1]
    explicit = {"assignment": "explicit", "stas": [3, 9]}
    assert [assigned_sta(explicit, i, 4) for i in range(4)] == [3, 9, 3, 9]


def test_prepare_emits_only_manual_cmdp_v2_fields(tmp_path):
    source = ROOT / "configs/mac_cmdp_experiment.yaml"
    spec = yaml.safe_load(source.read_text(encoding="utf-8"))
    generated_path = prepare(spec, source, paths(spec, tmp_path / "experiment"))
    reward = load_config(generated_path)["reward"]
    assert reward["version"] == "tgax_cmdp_v2"
    assert reward["cost_order"] == ["delay", "loss", "fairness", "starvation"]
    assert reward["budgets"] == [0.30, 0.15, 0.0, 0.25]
    assert reward["budget_status"] == "manual; no Default calibration"
    assert "low_priority_cost_budget" not in reward
    assert "low_priority_min_satisfaction" not in reward
    assert "minimum_timely_delivery" not in reward

