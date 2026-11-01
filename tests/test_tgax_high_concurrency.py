from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_tgax_load_variants import (
    assign_full40_multiac_flows,
    assign_full40_multiac_v2_flows,
    assign_full40_multiac_v3_flows,
    assign_full40_multiac_v4_flows,
    assign_full40_multiac_v5_flows,
    assign_full40_multiac_v6_flows,
)


def test_full40_multiac_coverage_and_counts() -> None:
    rows = assign_full40_multiac_flows(list(range(7)), 40, 20261021)
    assert len(rows) == 7 * 106
    for bss in range(7):
        group = [row for row in rows if row["bss"] == bss]
        high_ul = {row["sta"] for row in group
                   if row["direction"] == "UL" and row["ac"] in {"VO", "VI"}}
        be = {row["sta"] for row in group if row["ac"] == "BE"}
        bk = {row["sta"] for row in group if row["ac"] == "BK"}
        voip = {row["sta"] for row in group if row["service"] == "D6"}
        gaming_ul = {row["sta"] for row in group if row["service"] == "U3"}
        assert high_ul == set(range(40))
        assert len(be) == 32
        assert len(bk) == 16
        assert voip.isdisjoint(gaming_ul)
        assert len(voip) == 12
        assert len(gaming_ul) == 28


def test_checked_in_topology_still_has_7_by_40_stas() -> None:
    path = Path(__file__).resolve().parents[1] / "results/raw/tgax_s3_inputs_20261021/subset7/topology.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert [sum(row["role"] == "STA" and int(row["bss"]) == bss for row in rows)
            for bss in range(7)] == [40] * 7


def test_full40_multiac_v2_preserves_coverage_and_persistent_bk() -> None:
    rows = assign_full40_multiac_v2_flows(list(range(7)), 40, 20261021)
    assert len(rows) == 7 * 110
    for bss in range(7):
        group = [row for row in rows if row["bss"] == bss]
        high_ul = {row["sta"] for row in group
                   if row["direction"] == "UL" and row["ac"] in {"VO", "VI"}}
        bk = {row["sta"] for row in group if row["ac"] == "BK"}
        bk_gaming = [row for row in group if row["ac"] == "BK" and row["model"] == "GMG"]
        bk_ftp = [row for row in group if row["ac"] == "BK" and row["model"] == "FTP"]
        assert high_ul == set(range(40))
        assert len(bk) == 16
        assert len(bk_gaming) == 16
        assert len(bk_ftp) == 4


def test_full40_multiac_v3_preserves_coverage_with_random_bk_models() -> None:
    rows = assign_full40_multiac_v3_flows(list(range(7)), 40, 20261021)
    assert len(rows) == 7 * 114
    for bss in range(7):
        group = [row for row in rows if row["bss"] == bss]
        high_ul = {row["sta"] for row in group
                   if row["direction"] == "UL" and row["ac"] in {"VO", "VI"}}
        be = {row["sta"] for row in group if row["ac"] == "BE"}
        bk = {row["sta"] for row in group if row["ac"] == "BK"}
        bk_http = [row for row in group if row["ac"] == "BK" and row["model"] == "HTTP"]
        bk_ftp = [row for row in group if row["ac"] == "BK" and row["model"] == "FTP"]
        assert high_ul == set(range(40))
        assert len(be) == 32
        assert len(bk) == 16
        assert len(bk_http) == 16
        assert len(bk_ftp) == 8


def test_full40_multiac_v4_replaces_persistent_ul_gaming_with_voip() -> None:
    rows = assign_full40_multiac_v4_flows(list(range(7)), 40, 20261021)
    assert len(rows) == 7 * 114
    for bss in range(7):
        group = [row for row in rows if row["bss"] == bss]
        high_ul = {row["sta"] for row in group
                   if row["direction"] == "UL" and row["ac"] in {"VO", "VI"}}
        extra_voip = [row for row in group if row["service"] == "U6"]
        assert high_ul == set(range(40))
        assert len(extra_voip) == 28
        assert all(row["direction"] == "UL" and row["model"] == "VOIP"
                   and row["ac"] == "VO" for row in extra_voip)
        assert not any(row["service"] == "U3" for row in group)


def test_full40_multiac_v5_retains_all_models_without_ap_video_saturation() -> None:
    rows = assign_full40_multiac_v5_flows(list(range(7)), 40, 20261021)
    assert len(rows) == 7 * 110
    for bss in range(7):
        group = [row for row in rows if row["bss"] == bss]
        assert {row["sta"] for row in group
                if row["direction"] == "UL" and row["ac"] in {"VO", "VI"}} == set(range(40))
        assert len([row for row in group if row["service"] == "D5"]) == 1
        assert not any(row["service"] == "D2" for row in group)
        assert {row["model"] for row in group} == {"VOIP", "GMG", "BV3", "HTTP", "FTP"}


def test_full40_multiac_v6_has_one_bidir_and_forty_voip_uplinks() -> None:
    rows = assign_full40_multiac_v6_flows(list(range(7)), 40, 20261021)
    assert len(rows) == 7 * 99
    for bss in range(7):
        group = [row for row in rows if row["bss"] == bss]
        voip_ul = [row for row in group if row["model"] == "VOIP" and row["direction"] == "UL"]
        voip_dl = [row for row in group if row["model"] == "VOIP" and row["direction"] == "DL"]
        assert {row["sta"] for row in voip_ul} == set(range(40))
        assert len(voip_dl) == 1
        assert len({row["sta"] for row in group if row["ac"] == "BE"}) == 32
        assert len({row["sta"] for row in group if row["ac"] == "BK"}) == 16
