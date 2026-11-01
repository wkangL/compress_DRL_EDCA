"""Build deterministic TGax-derived topology and per-BSS flow assignments.

The full reuse-3 performance drop contains 19 active co-channel BSSs embedded
in the 61-cell reference area described by IEEE 802.11-14/0980r16.  The 7-BSS
input is an explicit center-plus-first-ring subset of that same drop.
The primary research input is a 3-BSS, 10-STA-per-BSS simplification that
retains the official geometry, PHY/MAC and one-quarter service mix.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import struct
import sys
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config


RADIUS_M = 10.0
BASE_ICD_M = math.sqrt(3.0) * RADIUS_M
AP_HEIGHT_M = 3.0
STA_HEIGHT_M = 1.5
FC_GHZ = 5.0
SHADOW_STD_DB = 5.0


def axial_disc(radius: int) -> list[tuple[int, int]]:
    return [(q, r) for q in range(-radius, radius + 1)
            for r in range(max(-radius, -q - radius), min(radius, -q + radius) + 1)]


def axial_xy(q: int, r: int) -> tuple[float, float]:
    # Pointy-top hexagons; nearest base-site separation is sqrt(3)*R.
    return BASE_ICD_M * (q + r / 2.0), 1.5 * RADIUS_M * r


def cochannel_to_base(q: int, r: int) -> tuple[int, int]:
    # Reuse-3 lattice basis.  Either basis vector has length 3R = 30 m.
    return q - r, q + 2 * r


def position_key(a: tuple[float, float, float], b: tuple[float, float, float]) -> bytes:
    left = tuple(round(v * 1000) for v in a)
    right = tuple(round(v * 1000) for v in b)
    if right < left:
        left, right = right, left
    return struct.pack("<6q", *(left + right))


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def link_shadow_db(a: tuple[float, float, float], b: tuple[float, float, float], seed: int) -> float:
    # Stable digest + SplitMix + Box-Muller.  The C++ propagation model uses the
    # same millimetre-coordinate key and transform.
    state = 0xCBF29CE484222325 ^ (seed & 0xFFFFFFFFFFFFFFFF)
    for byte in position_key(a, b):
        state ^= byte
        state = (state * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    x1, x2 = splitmix64(state), splitmix64(state ^ 0xD1B54A32D192ED03)
    u1 = (x1 + 0.5) / 2**64
    u2 = (x2 + 0.5) / 2**64
    return SHADOW_STD_DB * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def pathloss_db(distance_m: float) -> float:
    d = max(distance_m, 1.0)
    return (40.05 + 20 * math.log10(FC_GHZ / 2.4)
            + 20 * math.log10(min(d, 10.0))
            + (35 * math.log10(d / 10.0) if d > 10.0 else 0.0))


def inside_pointy_hex(x: float, y: float) -> bool:
    ax, ay = abs(x), abs(y)
    if ay > RADIUS_M or ax > math.sqrt(3) * RADIUS_M / 2:
        return False
    return ay <= RADIUS_M / 2 or ax <= math.sqrt(3) * (RADIUS_M - ay)


def uniform_hex(rng: random.Random) -> tuple[float, float]:
    while True:
        x = rng.uniform(-math.sqrt(3) * RADIUS_M / 2, math.sqrt(3) * RADIUS_M / 2)
        y = rng.uniform(-RADIUS_M, RADIUS_M)
        if inside_pointy_hex(x, y):
            return x, y


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def generate_full(seed: int, sta_per_bss: int) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    references = []
    for index, (q, r) in enumerate(sorted(axial_disc(4), key=lambda p: (max(abs(p[0]), abs(p[1]), abs(-p[0]-p[1])), p[1], p[0]))):
        x, y = axial_xy(q, r)
        references.append(dict(ref_id=index, base_q=q, base_r=r, x=x, y=y, z=AP_HEIGHT_M))

    active_axial = sorted(axial_disc(2), key=lambda p: (max(abs(p[0]), abs(p[1]), abs(-p[0]-p[1])), p[1], p[0]))
    active = []
    ref_by_axial = {(a["base_q"], a["base_r"]): a for a in references}
    for bss, (cq, cr) in enumerate(active_axial):
        bq, br = cochannel_to_base(cq, cr)
        ref = ref_by_axial[(bq, br)]
        active.append(dict(bss=bss, co_q=cq, co_r=cr, **ref))

    accepted: dict[int, list[dict]] = {a["bss"]: [] for a in active}
    active_by_ref = {a["ref_id"]: a for a in active}
    candidate = 0
    while any(len(values) < sta_per_bss for values in accepted.values()):
        base = references[rng.randrange(len(references))]
        dx, dy = uniform_hex(rng)
        pos = (base["x"] + dx, base["y"] + dy, STA_HEIGHT_M)
        powers = []
        for ap in references:
            ap_pos = (ap["x"], ap["y"], ap["z"])
            shadow = link_shadow_db(pos, ap_pos, seed)
            loss = pathloss_db(distance(pos, ap_pos)) + shadow
            powers.append((20.0 - loss, ap["ref_id"], loss, shadow))
        powers.sort(reverse=True)
        best_power, best_ref, loss, shadow = powers[0]
        target = active_by_ref.get(best_ref)
        if target is not None and len(accepted[target["bss"]]) < sta_per_bss:
            accepted[target["bss"]].append(dict(
                candidate_id=candidate, x=pos[0], y=pos[1], z=pos[2],
                strongest_ref=best_ref, strongest_rx_dbm=best_power,
                serving_pathloss_shadow_db=loss, serving_shadow_db=shadow,
                second_rx_dbm=powers[1][0], strongest_margin_db=best_power - powers[1][0]))
        candidate += 1
        if candidate > 2_000_000:
            raise RuntimeError("STA rejection sampling failed to fill all active BSSs")

    nodes: list[dict] = []
    for ap in active:
        nodes.append(dict(node_label=f"ap{ap['bss']}", role="AP", bss=ap["bss"], sta=-1,
                          x=ap["x"], y=ap["y"], z=ap["z"], channel=42,
                          strongest_ref=ap["ref_id"], strongest_rx_dbm="",
                          second_rx_dbm="", strongest_margin_db=""))
        for sta, item in enumerate(accepted[ap["bss"]]):
            nodes.append(dict(node_label=f"b{ap['bss']}_sta{sta}", role="STA", bss=ap["bss"], sta=sta,
                              x=item["x"], y=item["y"], z=item["z"], channel=42,
                              strongest_ref=item["strongest_ref"], strongest_rx_dbm=item["strongest_rx_dbm"],
                              second_rx_dbm=item["second_rx_dbm"], strongest_margin_db=item["strongest_margin_db"]))
    return active, references, nodes


SERVICES = {
    "D1": ("BV6", "DL", "TCP", "VI", 12),
    "D2": ("BV3", "DL", "TCP", "VI", 8),
    "D3": ("FTP", "DL", "TCP", "BE", 4),
    "D4": ("HTTP", "DL", "TCP", "BE", 12),
    "D5": ("GMG", "DL", "UDP", "VI", 16),
    "D6": ("VOIP", "BIDIR", "UDP", "VO", 12),
    "U1": ("BV3", "UL", "TCP", "VI", 4),
    "U2": ("FTP", "UL", "TCP", "BE", 4),
    "U3": ("GMG", "UL", "UDP", "VI", 16),
}


def assign_flows(bss_ids: Iterable[int], sta_per_bss: int, seed: int,
                 service_divisor: int = 1,
                 service_counts: dict[str, int] | None = None) -> list[dict]:
    rows: list[dict] = []
    flow_id = 0
    for bss in bss_ids:
        for service_index, (service, (model, direction, transport, ac, official_count)) in enumerate(SERVICES.items()):
            if service_counts is None:
                if official_count % service_divisor:
                    raise ValueError(f"{service} count {official_count} is not divisible by {service_divisor}")
                count = official_count // service_divisor
            else:
                count = service_counts[service]
            if count > sta_per_bss:
                raise ValueError(f"{service} count {count} exceeds {sta_per_bss} distinct STAs")
            rng = random.Random(seed * 100_003 + bss * 101 + service_index)
            chosen = rng.sample(range(sta_per_bss), count)
            for assignment, sta in enumerate(chosen):
                directions = ("DL", "UL") if direction == "BIDIR" else (direction,)
                for actual in directions:
                    rows.append(dict(flow_id=flow_id, bss=bss, service=service, model=model,
                                     assignment=assignment, direction=actual, transport=transport,
                                     ac=ac, sta=sta,
                                     source=f"ap{bss}" if actual == "DL" else f"b{bss}_sta{sta}",
                                     sink=f"b{bss}_sta{sta}" if actual == "DL" else f"ap{bss}"))
                    flow_id += 1
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def subset_nodes(nodes: list[dict], bss_ids: set[int], sta_limit: int | None = None) -> list[dict]:
    return [row for row in nodes
            if row["bss"] in bss_ids and
            (row["role"] == "AP" or sta_limit is None or int(row["sta"]) < sta_limit)]


def build(seed: int, output: Path) -> None:
    config = load_config(ROOT / "configs/tgax_s3_full19.json")
    sta_per_bss = int(config["layout"]["sta_per_bss"])
    active, references, nodes = generate_full(seed, sta_per_bss)
    full_ids = [a["bss"] for a in active]
    subset_ids = {a["bss"] for a in active if max(abs(a["co_q"]), abs(a["co_r"]), abs(-a["co_q"]-a["co_r"])) <= 1}
    center_ids = {a["bss"] for a in active if a["co_q"] == 0 and a["co_r"] == 0}
    main3_ids = {0, 1, 2}
    if len(active) != 19 or len(subset_ids) != 7:
        raise AssertionError("Unexpected reuse-3 lattice cardinality")

    for name, ids in (("full19", set(full_ids)), ("subset7", subset_ids),
                      ("api1", center_ids), ("main3", main3_ids)):
        variant_config = load_config(ROOT / "configs" /
                                     ({"full19": "tgax_s3_full19.json",
                                       "subset7": "tgax_s3_7bss.json",
                                       "api1": "tgax_s3_1bss_api_smoke.json",
                                       "main3": "tgax_s3_3ap_10sta.json"}[name]))
        root = output / name
        variant_sta_per_bss = int(variant_config["layout"]["sta_per_bss"])
        selected_active = [a for a in active if a["bss"] in ids]
        selected_nodes = subset_nodes(nodes, ids, variant_sta_per_bss)
        divisor = 4 if name == "main3" else 1
        flows = assign_flows(sorted(ids), variant_sta_per_bss, seed, divisor)
        # One representative of every service and direction in every BSS.  This
        # is a protocol smoke input only; it is never used as a performance
        # substitute for the complete official traffic assignment above.
        seen: set[tuple[int, str, str]] = set()
        smoke_flows: list[dict] = []
        for flow in flows:
            key = (flow["bss"], flow["service"], flow["direction"])
            if key not in seen:
                seen.add(key)
                smoke_flows.append(flow)
        write_csv(root / "topology.csv", selected_nodes)
        write_csv(root / "flows.csv", flows)
        write_csv(root / "protocol_smoke_flows.csv", smoke_flows)
        write_csv(root / "active_bss.csv", selected_active)
        write_csv(root / "reference_61_cells.csv", references)
        resolved = dict(config=variant_config, seed=seed, variant=name,
                        active_bss=len(ids), sta_count=len(ids) * variant_sta_per_bss,
                        effective_service_counts_per_bss={
                            service: values[4] // divisor for service, values in SERVICES.items()
                        },
                        one_way_application_components_per_bss=len(flows) // len(ids),
                        topology_csv=str((root / "topology.csv").resolve()),
                        flows_csv=str((root / "flows.csv").resolve()),
                        official_complete_scenario=(name == "full19"),
                        note=({"full19": "all 19 cochannel BSSs",
                               "subset7": "center plus first cochannel ring; official-scene subset",
                               "api1": "debug-only center BSS API smoke variant",
                               "main3": "primary TGax-parameter-based simplified 3-BSS, 10-STA-per-BSS scene"}[name]))
        (root / "resolved_config.json").write_text(json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/raw/tgax_s3_inputs_20261021")
    args = parser.parse_args()
    build(args.seed, args.output)


if __name__ == "__main__":
    main()
