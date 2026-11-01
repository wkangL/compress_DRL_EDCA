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
    assert spec["training"]["parallel_workers"] == 1
    assert spec["training"]["stages"] * spec["training"]["episode_s"] == 100
    assert spec["simulation"]["training_seeds"] == [42]
    assert spec["simulation"]["evaluation_seeds"] == [66]


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


def test_prepare_emits_only_manual_two_cost_fields(tmp_path):
    source = ROOT / "configs/mac_cmdp_experiment.yaml"
    spec = yaml.safe_load(source.read_text(encoding="utf-8"))
    generated_path = prepare(spec, source, paths(spec, tmp_path / "experiment"))
    reward = load_config(generated_path)["reward"]
    assert reward["version"] == "tgax_cmdp_two_cost_v4"
    assert reward["cost_order"] == ["qos", "starvation"]
    assert reward["budgets"] == [0.40, 0.90]
    assert reward["multiplier_initial"] == [0.1, 0.1]
    assert reward["qos_delay_reference_ms"] == [20.0, 50.0]
    assert reward["budget_status"] == "manual; no Default calibration"
    assert "low_priority_cost_budget" not in reward
    assert "low_priority_min_satisfaction" not in reward
    assert "minimum_timely_delivery" not in reward
    assert "delay_excess_cap" not in reward
    assert "minimum_jain" not in reward
    assert "minimum_satisfaction" not in reward


def test_immediate_mac_admission_does_not_schedule_polling():
    source = (ROOT / "ns3_ext/tgax-control.cc").read_text(encoding="utf-8")
    start = source.index("void StartApplication() override", source.index("class MacArrivalApplication"))
    stop = source.index("void StopApplication() override", start)
    startup = source[start:stop]
    assert "if(m_highWatermark>0)" in startup
    assert startup.index("if(m_highWatermark>0)") < startup.index("&MacArrivalApplication::AdmissionTick")

    tick_start = source.index("void AdmissionTick()", stop)
    tick_end = source.index("void VoipTick()", tick_start)
    tick = source[tick_start:tick_end]
    assert "if(!m_running || m_highWatermark==0) return;" in tick

