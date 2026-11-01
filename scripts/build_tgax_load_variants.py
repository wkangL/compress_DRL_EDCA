"""Build official and TGax-based load-count extensions for a TGax scene.

Only the number of independent application instances changes.  Each instance
continues to use the same packet-size, inter-arrival, state-duration and
TCP/UDP generator implemented by tgax-s3.cc.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from fractions import Fraction
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT
from scripts.build_tgax_s3_inputs import SERVICES, write_csv


VARIANTS: dict[str, tuple[Fraction, bool, str]] = {
    "load025": (Fraction(1, 4), False, "25% of every official service count"),
    "load050": (Fraction(1, 2), False, "50% of every official service count"),
    "load075": (Fraction(3, 4), False, "75% of every official service count"),
    "load100": (Fraction(1, 1), False, "100% official TGax service counts"),
    "load125": (Fraction(5, 4), False, "125% of every official service count"),
    "load150": (Fraction(3, 2), False, "150% of every official service count"),
    "rt125": (Fraction(5, 4), True, "BE fixed at 100%; VO/VI service counts at 125%"),
    "rt150": (Fraction(3, 2), True, "BE fixed at 100%; VO/VI service counts at 150%"),
}

SPECIAL_VARIANTS: dict[str, tuple[int | dict[str, int], str, dict[str, str], bool]] = {
    "mix_min1": (1, "one instance of every official service row per BSS", {}, False),
    "mix_min2": (2, "two instances of every official service row per BSS", {}, False),
    "candidate_ftp_bk": (1, "mix_min1 with FTP D3/U2 mapped to BK", {"D3": "BK", "U2": "BK"}, False),
    "candidate_voip8_gaming4_ftp_bk": (
        {"D1": 1, "D2": 1, "D3": 1, "D4": 1, "D5": 4,
         "D6": 8, "U1": 1, "U2": 1, "U3": 4},
        "candidate_ftp_bk with eight bidirectional VoIP pairs and four DL/UL Gaming flows per BSS",
        {"D3": "BK", "U2": "BK"}, True),
    "image_business_40sta": (
        {"D1": 12, "D2": 5, "D3": 12, "D4": 28, "D5": 28,
         "D6": 24, "U1": 5, "U2": 8, "U3": 40},
        "user-specified 40-STA mix: VoIP 24/24, BV6 12 DL, BV3 5/5, "
        "Gaming 28/40, HTTP 28 DL, and FTP 12/8 per BSS",
        {"D3": "BK", "U2": "BK"}, True),
}

HIGH_CONCURRENCY_VARIANT = "candidate_full40_multiac_v1"
HIGH_CONCURRENCY_VARIANT_V2 = "candidate_full40_multiac_v2"
HIGH_CONCURRENCY_VARIANT_V3 = "candidate_full40_multiac_v3"
HIGH_CONCURRENCY_VARIANT_V4 = "candidate_full40_multiac_v4"
HIGH_CONCURRENCY_VARIANT_V5 = "candidate_full40_multiac_v5"
HIGH_CONCURRENCY_VARIANT_V6 = "candidate_full40_multiac_v6"


def assign_full40_multiac_flows(bss_ids: list[int], sta_per_bss: int, seed: int) -> list[dict]:
    """Build the reviewed 40-STA high-concurrency candidate.

    Twelve bidirectional VoIP users and 28 uplink Gaming users are disjoint,
    hence every associated STA has an uplink VO or VI source. BE and BK use
    separate deterministic permutations to obtain broad coverage. Only the
    number, AC and endpoint assignment change; model names remain those
    consumed by the existing TGax generators.
    """
    if sta_per_bss != 40:
        raise ValueError(f"{HIGH_CONCURRENCY_VARIANT} requires exactly 40 STAs per BSS")
    specifications = (
        # label, model, direction, transport, AC, count
        ("D6", "VOIP", "BIDIR", "UDP", "VO", 12),
        ("U3", "GMG", "UL", "UDP", "VI", 28),
        ("D5", "GMG", "DL", "UDP", "VI", 4),
        ("D2", "BV3", "DL", "TCP", "VI", 1),
        ("U1", "BV3", "UL", "TCP", "VI", 1),
        ("D4", "HTTP", "DL", "TCP", "BE", 20),
        ("U2_BE", "FTP", "UL", "TCP", "BE", 12),
        ("U2_BK", "FTP", "UL", "TCP", "BK", 8),
        ("D3_BK", "FTP", "DL", "TCP", "BK", 8),
    )
    result: list[dict] = []
    flow_id = 0
    for bss in bss_ids:
        high = list(range(sta_per_bss))
        random.Random(seed * 100_003 + bss * 101 + 91_001).shuffle(high)
        low = list(range(sta_per_bss))
        random.Random(seed * 100_003 + bss * 101 + 91_002).shuffle(low)
        bk = list(range(sta_per_bss))
        random.Random(seed * 100_003 + bss * 101 + 91_003).shuffle(bk)
        generic = list(range(sta_per_bss))
        random.Random(seed * 100_003 + bss * 101 + 91_004).shuffle(generic)
        assignments = {
            "D6": high[:12],
            "U3": high[12:],
            "D5": generic[:4],
            "D2": generic[4:5],
            "U1": generic[5:6],
            "D4": low[:20],
            "U2_BE": low[20:32],
            "U2_BK": bk[:8],
            "D3_BK": bk[8:16],
        }
        for service_index, (service, model, direction, transport, ac, count) in enumerate(specifications):
            chosen = assignments[service]
            if len(chosen) != count or len(set(chosen)) != count:
                raise AssertionError(f"Invalid {service} assignment in BSS {bss}")
            for assignment, sta in enumerate(chosen):
                directions = ("DL", "UL") if direction == "BIDIR" else (direction,)
                for actual in directions:
                    stream_id = bss * 10_000 + service_index * 100 + assignment * 2 + (actual == "UL")
                    result.append(dict(
                        flow_id=flow_id, stream_id=int(stream_id), bss=bss,
                        service=service, model=model, assignment=assignment,
                        direction=actual, transport=transport, ac=ac, sta=sta,
                        source=f"ap{bss}" if actual == "DL" else f"b{bss}_sta{sta}",
                        sink=f"b{bss}_sta{sta}" if actual == "DL" else f"ap{bss}",
                    ))
                    flow_id += 1
    return result


def assign_full40_multiac_v2_flows(bss_ids: list[int], sta_per_bss: int, seed: int) -> list[dict]:
    """Replace FTP-only BK load with persistent TGax Gaming-shaped BK load.

    The v1 gate showed zero successful BK MPDUs in BSS0.  V2 retains two FTP
    sessions per direction but uses eight UL and eight DL instances of the
    unmodified Gaming generator as the persistent BK pressure.  Endpoint
    coverage is unchanged and no packet-size or inter-arrival distribution is
    altered.
    """
    rows = assign_full40_multiac_flows(bss_ids, sta_per_bss, seed)
    result: list[dict] = []
    removed: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["service"] in {"U2_BK", "D3_BK"}:
            removed[(row["bss"], row["service"])].append(row)
            converted = dict(row)
            converted.update(
                service="U3_BK" if row["direction"] == "UL" else "D5_BK",
                model="GMG", transport="UDP",
            )
            result.append(converted)
        else:
            result.append(dict(row))
    for bss in bss_ids:
        for direction, source_service, new_service, stream_base in (
            ("UL", "U2_BK", "U2_BK_FTP", 900),
            ("DL", "D3_BK", "D3_BK_FTP", 950),
        ):
            for assignment, source in enumerate(removed[(bss, source_service)][:2]):
                sta = int(source["sta"])
                result.append(dict(
                    flow_id=-1, stream_id=bss * 10_000 + stream_base + assignment * 2 + (direction == "UL"),
                    bss=bss, service=new_service, model="FTP", assignment=assignment,
                    direction=direction, transport="TCP", ac="BK", sta=sta,
                    source=f"ap{bss}" if direction == "DL" else f"b{bss}_sta{sta}",
                    sink=f"b{bss}_sta{sta}" if direction == "DL" else f"ap{bss}",
                ))
    for flow_id, row in enumerate(result):
        row["flow_id"] = flow_id
    return result


def assign_full40_multiac_v3_flows(bss_ids: list[int], sta_per_bss: int, seed: int) -> list[dict]:
    """Final balanced candidate after the v2 20-second Default overload gate.

    V2's persistent Gaming-shaped BK traffic caused 97.5% final failure.  V3
    preserves the same 16-STA BK coverage but uses 16 unmodified HTTP downlink
    instances and eight unmodified FTP uplink instances.  This keeps real BK
    activity in a 20-second window without changing either generator's random
    distributions, and removes the artificial persistent UDP flood.
    """
    rows = assign_full40_multiac_flows(bss_ids, sta_per_bss, seed)
    result = [dict(row) for row in rows if row["service"] not in {"U2_BK", "D3_BK"}]
    by_bss: dict[int, list[int]] = defaultdict(list)
    ftp_by_bss: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        if row["service"] in {"U2_BK", "D3_BK"} and int(row["sta"]) not in by_bss[int(row["bss"])]:
            by_bss[int(row["bss"])].append(int(row["sta"]))
        if row["service"] == "U2_BK":
            ftp_by_bss[int(row["bss"])].append(int(row["sta"]))
    for bss in bss_ids:
        if len(by_bss[bss]) != 16 or len(ftp_by_bss[bss]) != 8:
            raise AssertionError(f"Invalid final BK coverage in BSS {bss}")
        for assignment, sta in enumerate(by_bss[bss]):
            result.append(dict(
                flow_id=-1, stream_id=bss * 10_000 + 1000 + assignment * 2,
                bss=bss, service="D4_BK_HTTP", model="HTTP", assignment=assignment,
                direction="DL", transport="TCP", ac="BK", sta=sta,
                source=f"ap{bss}", sink=f"b{bss}_sta{sta}",
            ))
        for assignment, sta in enumerate(ftp_by_bss[bss]):
            result.append(dict(
                flow_id=-1, stream_id=bss * 10_000 + 1100 + assignment * 2 + 1,
                bss=bss, service="U2_BK_FTP", model="FTP", assignment=assignment,
                direction="UL", transport="TCP", ac="BK", sta=sta,
                source=f"b{bss}_sta{sta}", sink=f"ap{bss}",
            ))
    for flow_id, row in enumerate(result):
        row["flow_id"] = flow_id
    return result


def assign_full40_multiac_v4_flows(bss_ids: list[int], sta_per_bss: int, seed: int) -> list[dict]:
    """Final candidate: retain 40 high-priority UL users without UDP flooding.

    The twelve bidirectional VoIP users are unchanged.  The other 28 users use
    independent uplink instances of the same unmodified TGax VoIP generator
    instead of persistent Gaming UL.  VI remains active through four downlink
    Gaming flows and the bidirectional BV3 pair.  BE/BK are inherited from v3.
    """
    rows = assign_full40_multiac_v3_flows(bss_ids, sta_per_bss, seed)
    for row in rows:
        if row["service"] == "U3":
            row.update(service="U6", model="VOIP", transport="UDP", ac="VO")
    return rows


def assign_full40_multiac_v5_flows(bss_ids: list[int], sta_per_bss: int, seed: int) -> list[dict]:
    """Final gate: remove the continuously backlogged AP-side VI bottleneck.

    V4 showed that sparse BK still received no service while the AP carried a
    BV3 downlink and four Gaming downlinks.  V5 retains one Gaming downlink and
    the BV3 uplink, so every TGax application model remains represented, while
    avoiding an always-backlogged AP VI queue that deterministically starves
    its colocated BK queue through EDCA virtual collisions.
    """
    rows = assign_full40_multiac_v4_flows(bss_ids, sta_per_bss, seed)
    result = []
    kept_d5: set[int] = set()
    for row in rows:
        if row["service"] == "D2":
            continue
        if row["service"] == "D5":
            bss = int(row["bss"])
            if bss in kept_d5:
                continue
            kept_d5.add(bss)
        result.append(dict(row))
    for flow_id, row in enumerate(result):
        row["flow_id"] = flow_id
    return result


def assign_full40_multiac_v6_flows(bss_ids: list[int], sta_per_bss: int, seed: int) -> list[dict]:
    """Sole formal candidate: one bidirectional and 39 uplink VoIP users.

    This preserves one high-priority uplink source on every STA while removing
    eleven extra AP-side VO streams whose colocated virtual collisions left too
    few BK successes for a meaningful delay estimate.
    """
    rows = assign_full40_multiac_v5_flows(bss_ids, sta_per_bss, seed)
    result = []
    for row in rows:
        if row["service"] == "D6" and row["direction"] == "DL" and int(row["assignment"]) > 0:
            continue
        item = dict(row)
        if item["service"] == "D6" and item["direction"] == "UL" and int(item["assignment"]) > 0:
            item["service"] = "U6_BASE"
        result.append(item)
    for flow_id, row in enumerate(result):
        row["flow_id"] = flow_id
    return result


def counts_for(scale: Fraction, realtime_only: bool) -> dict[str, int]:
    result: dict[str, int] = {}
    for service, (_, _, _, ac, official) in SERVICES.items():
        effective = Fraction(1, 1) if realtime_only and ac == "BE" else scale
        count = Fraction(official) * effective
        if count.denominator != 1:
            raise ValueError(f"Non-integer count for {service}: {count}")
        result[service] = int(count)
    return result


def assign_nested_flows(base_rows: list[dict[str, str]], bss_ids: list[int], sta_per_bss: int,
                        seed: int, counts: dict[str, int],
                        ac_overrides: dict[str, str] | None = None,
                        spread_new_instances: bool = False) -> list[dict]:
    """Keep lower loads as prefixes of the official STA assignment.

    Extra assignments above 100% use a deterministic permutation of STAs not
    selected by the official configuration.  stream_id is independent of the
    total number of rows, so a logical flow keeps its application RNG stream
    across load variants.
    """
    official: dict[tuple[int, str], list[int]] = {}
    for bss in bss_ids:
        for service in SERVICES:
            rows = [row for row in base_rows if int(row["bss"]) == bss and row["service"] == service]
            assignments = {int(row["assignment"]): int(row["sta"]) for row in rows}
            official[(bss, service)] = [assignments[index] for index in sorted(assignments)]
    result: list[dict] = []
    ac_overrides = ac_overrides or {}
    flow_id = 0
    for bss in bss_ids:
        chosen_by_service: dict[str, list[int]] = {}
        if spread_new_instances:
            # Preserve assignment zero from candidate_ftp_bk exactly, then place
            # only added instances on idle STAs whenever an official candidate is
            # available.  This maximizes independently contending STAs without
            # changing the topology or any application generator internals.
            used = {official[(bss, service)][0] for service in SERVICES}
            for service in SERVICES:
                chosen = [official[(bss, service)][0]]
                candidates = official[(bss, service)][1:]
                remaining = [sta for sta in range(sta_per_bss)
                             if sta not in official[(bss, service)]]
                random.Random(seed * 100_003 + bss * 101 + list(SERVICES).index(service) + 9_000_001).shuffle(remaining)
                candidates = candidates + remaining
                while len(chosen) < counts[service]:
                    unused = next((sta for sta in candidates if sta not in used and sta not in chosen), None)
                    sta = unused if unused is not None else next(sta for sta in candidates if sta not in chosen)
                    chosen.append(sta)
                    used.add(sta)
                chosen_by_service[service] = chosen
        for service_index, (service, (model, direction, transport, ac, _)) in enumerate(SERVICES.items()):
            if spread_new_instances:
                chosen = chosen_by_service[service]
            else:
                chosen = list(official[(bss, service)])
                remaining = [sta for sta in range(sta_per_bss) if sta not in chosen]
                random.Random(seed * 100_003 + bss * 101 + service_index + 9_000_001).shuffle(remaining)
                chosen.extend(remaining)
            for assignment, sta in enumerate(chosen[:counts[service]]):
                directions = ("DL", "UL") if direction == "BIDIR" else (direction,)
                for actual in directions:
                    stream_id = bss * 10_000 + service_index * 100 + assignment * 2 + (actual == "UL")
                    result.append(dict(flow_id=flow_id, stream_id=int(stream_id), bss=bss,
                                       service=service, model=model, assignment=assignment,
                                       direction=actual, transport=transport,
                                       ac=ac_overrides.get(service, ac), sta=sta,
                                       source=f"ap{bss}" if actual == "DL" else f"b{bss}_sta{sta}",
                                       sink=f"b{bss}_sta{sta}" if actual == "DL" else f"ap{bss}"))
                    flow_id += 1
    return result


def build(inputs_root: Path, seed: int, scene: str) -> None:
    scene_root = inputs_root / scene
    if not (scene_root / "topology.csv").is_file():
        raise FileNotFoundError(f"Build base TGax inputs first: {scene_root}")
    with (scene_root / "topology.csv").open(newline="", encoding="utf-8") as stream:
        topology = list(csv.DictReader(stream))
    with (scene_root / "flows.csv").open(newline="", encoding="utf-8") as stream:
        base_rows = list(csv.DictReader(stream))
    bss_ids = sorted({int(row["bss"]) for row in topology if row["role"] == "AP"})
    sta_per_bss = min(sum(row["role"] == "STA" and int(row["bss"]) == bss for row in topology)
                      for bss in bss_ids)
    expected_bss = {"subset7": 7, "full19": 19}.get(scene)
    if expected_bss is None:
        raise ValueError(f"Unsupported scene: {scene}")
    if len(bss_ids) != expected_bss or sta_per_bss != 40:
        raise ValueError(f"Expected {expected_bss} BSS x 40 STA, got {len(bss_ids)} x {sta_per_bss}")

    catalog = []
    definitions = [(name, scale, realtime_only, description, None, {})
                   for name, (scale, realtime_only, description) in VARIANTS.items()]
    definitions += [(name, None, False, description, count_spec, overrides, spread)
                    for name, (count_spec, description, overrides, spread) in SPECIAL_VARIANTS.items()]
    definitions = [(*row, False) if len(row) == 6 else row for row in definitions]
    definitions.append((HIGH_CONCURRENCY_VARIANT, None, False,
                        "40-STA uplink VO/VI coverage with broad BE/BK concurrency",
                        None, {}, False))
    definitions.append((HIGH_CONCURRENCY_VARIANT_V2, None, False,
                        "v1 corrected after the Default gate: persistent Gaming-shaped BK plus retained FTP",
                        None, {}, False))
    definitions.append((HIGH_CONCURRENCY_VARIANT_V3, None, False,
                        "final candidate: v2 BK overload corrected with broad HTTP/FTP BK coverage",
                        None, {}, False))
    definitions.append((HIGH_CONCURRENCY_VARIANT_V4, None, False,
                        "final candidate: 40-STA high-priority UL coverage with VoIP-shaped UL and balanced HTTP/FTP BK",
                        None, {}, False))
    definitions.append((HIGH_CONCURRENCY_VARIANT_V5, None, False,
                        "final gate: broad multi-AC coverage without a continuously backlogged AP-side VI source",
                        None, {}, False))
    definitions.append((HIGH_CONCURRENCY_VARIANT_V6, None, False,
                        "sole formal candidate: one bidirectional plus 39 uplink VoIP users and broad BE/BK coverage",
                        None, {}, False))
    for name, scale, realtime_only, description, count_spec, ac_overrides, spread in definitions:
        if name in {HIGH_CONCURRENCY_VARIANT, HIGH_CONCURRENCY_VARIANT_V2,
                    HIGH_CONCURRENCY_VARIANT_V3, HIGH_CONCURRENCY_VARIANT_V4,
                    HIGH_CONCURRENCY_VARIANT_V5, HIGH_CONCURRENCY_VARIANT_V6}:
            if name == HIGH_CONCURRENCY_VARIANT:
                flows = assign_full40_multiac_flows(bss_ids, sta_per_bss, seed)
                counts = {"D6": 12, "U3": 28, "D5": 4, "D2": 1, "U1": 1,
                          "D4": 20, "U2_BE": 12, "U2_BK": 8, "D3_BK": 8}
                bk_description = "16 FTP-only BK instances"
            elif name == HIGH_CONCURRENCY_VARIANT_V2:
                flows = assign_full40_multiac_v2_flows(bss_ids, sta_per_bss, seed)
                counts = {"D6": 12, "U3": 28, "D5": 4, "D2": 1, "U1": 1,
                          "D4": 20, "U2_BE": 12, "U3_BK": 8, "D5_BK": 8,
                          "U2_BK_FTP": 2, "D3_BK_FTP": 2}
                bk_description = "16 persistent Gaming-shaped and 4 retained FTP BK instances"
            elif name == HIGH_CONCURRENCY_VARIANT_V3:
                flows = assign_full40_multiac_v3_flows(bss_ids, sta_per_bss, seed)
                counts = {"D6": 12, "U3": 28, "D5": 4, "D2": 1, "U1": 1,
                          "D4": 20, "U2_BE": 12, "D4_BK_HTTP": 16,
                          "U2_BK_FTP": 8}
                bk_description = "16 HTTP DL plus 8 FTP UL BK instances across 16 unique STAs"
            elif name == HIGH_CONCURRENCY_VARIANT_V4:
                flows = assign_full40_multiac_v4_flows(bss_ids, sta_per_bss, seed)
                counts = {"D6": 12, "U6": 28, "D5": 4, "D2": 1, "U1": 1,
                          "D4": 20, "U2_BE": 12, "D4_BK_HTTP": 16,
                          "U2_BK_FTP": 8}
                bk_description = "16 HTTP DL plus 8 FTP UL BK instances across 16 unique STAs"
            elif name == HIGH_CONCURRENCY_VARIANT_V5:
                flows = assign_full40_multiac_v5_flows(bss_ids, sta_per_bss, seed)
                counts = {"D6": 12, "U6": 28, "D5": 1, "U1": 1,
                          "D4": 20, "U2_BE": 12, "D4_BK_HTTP": 16,
                          "U2_BK_FTP": 8}
                bk_description = "16 HTTP DL plus 8 FTP UL BK instances across 16 unique STAs"
            else:
                flows = assign_full40_multiac_v6_flows(bss_ids, sta_per_bss, seed)
                counts = {"D6_BIDIR": 1, "U6_BASE": 11, "U6": 28,
                          "D5": 1, "U1": 1, "D4": 20, "U2_BE": 12,
                          "D4_BK_HTTP": 16, "U2_BK_FTP": 8}
                bk_description = "16 HTTP DL plus 8 FTP UL BK instances across 16 unique STAs"
            target = scene_root / name
            write_csv(target / "flows.csv", flows)
            manifest = {
                "name": name,
                "classification": "TGax-based research high-concurrency scenario",
                "description": description,
                "seed": seed,
                "scene": scene,
                "active_bss": expected_bss,
                "sta_per_bss": 40,
                "load_mode": "custom_service_counts_and_ac_mapping",
                "service_counts_per_bss": counts,
                "one_way_components_per_bss": len(flows) // len(bss_ids),
                "generator_internals_changed": False,
                "stable_logical_rng_streams": True,
                "all_sta_have_uplink_vo_or_vi": True,
                "voip_and_uplink_gaming_disjoint": name in {
                    HIGH_CONCURRENCY_VARIANT, HIGH_CONCURRENCY_VARIANT_V2,
                    HIGH_CONCURRENCY_VARIANT_V3,
                },
                "uplink_high_priority_assignment": (
                    "one bidirectional VoIP user plus 39 uplink-only VoIP users"
                    if name == HIGH_CONCURRENCY_VARIANT_V6 else
                    "12 bidirectional VoIP users plus 28 disjoint uplink VoIP users"
                    if name in {HIGH_CONCURRENCY_VARIANT_V4, HIGH_CONCURRENCY_VARIANT_V5}
                    else "12 bidirectional VoIP users plus 28 disjoint uplink Gaming users"
                ),
                "be_unique_sta_per_bss": 32,
                "bk_unique_sta_per_bss": 16,
                "bk_load_design": bk_description,
                "ac_overrides": {"U2_BE": "BE", "U3_BK": "BK", "D5_BK": "BK",
                                 "D4_BK_HTTP": "BK",
                                 "U2_BK_FTP": "BK", "D3_BK_FTP": "BK"},
            }
            (target / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            catalog.append(manifest)
            continue
        counts = ({service: count_spec for service in SERVICES} if isinstance(count_spec, int)
                  else dict(count_spec) if isinstance(count_spec, dict)
                  else counts_for(scale, realtime_only))
        flows = assign_nested_flows(base_rows, bss_ids, sta_per_bss, seed, counts,
                                    ac_overrides, spread)
        target = scene_root / name
        write_csv(target / "flows.csv", flows)
        manifest = {
            "name": name,
            "classification": ("official" if name == "load100" else
                               "TGax-model-driven custom concurrency candidate"
                               if name == "candidate_voip8_gaming4_ftp_bk" else
                               "candidate research configuration" if name == "candidate_ftp_bk" else
                               "TGax-based extension scenario"),
            "description": description,
            "seed": seed,
            "scene": scene,
            "active_bss": expected_bss,
            "sta_per_bss": 40,
            "scale": float(scale) if scale is not None else None,
            "load_mode": ("custom_service_counts" if isinstance(count_spec, dict) else
                          "uniform_service_count" if isinstance(count_spec, int) else
                          "official_count_multiplier"),
            "realtime_only": realtime_only,
            "service_counts_per_bss": counts,
            "one_way_components_per_bss": len(flows) // len(bss_ids),
            "generator_internals_changed": False,
            "nested_assignment": True,
            "stable_logical_rng_streams": True,
            "spread_new_instances_across_idle_stas": spread,
            "ac_overrides": ac_overrides,
        }
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        catalog.append(manifest)
    (scene_root / "load_variant_catalog.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-root", type=Path,
                        default=ROOT / "results/raw/tgax_s3_inputs_20261021")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--scene", choices=("subset7", "full19"), default="subset7")
    args = parser.parse_args()
    build(args.inputs_root, args.seed, args.scene)


if __name__ == "__main__":
    main()
