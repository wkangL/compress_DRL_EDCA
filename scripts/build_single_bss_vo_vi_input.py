"""Build the user-requested single-BSS VO/VI-only TGax input."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_config import ROOT
from scripts.build_tgax_s3_inputs import write_csv


def main() -> None:
    base = ROOT / "results/raw/tgax_s3_inputs_20261021"
    source = base / "subset7/image_business_40sta/flows.csv"
    target = base / "api1/image_vo_vi_single_bss"
    with source.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)
                if int(row["bss"]) == 0 and row["ac"] in {"VO", "VI"}]
    for flow_id, row in enumerate(rows):
        row["flow_id"] = flow_id
    write_csv(target / "flows.csv", rows)
    shutil.copy2(base / "api1/topology.csv", target / "topology.csv")
    counts: dict[str, int] = {}
    for row in rows:
        key = f'{row["service"]}_{row["direction"]}'
        counts[key] = counts.get(key, 0) + 1
    manifest = {
        "name": "image_vo_vi_single_bss",
        "classification": "single-BSS prescribed-arrival VO/VI saturation check",
        "active_bss": 1,
        "sta_per_bss": 40,
        "interfering_bss": 0,
        "enabled_acs": ["VO", "VI"],
        "service_components": counts,
        "one_way_components": len(rows),
        "traffic_models_unchanged": True,
        "full_buffer_claim": "must be established from measured queue nonempty fractions",
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    mac_target = base / "api1/image_mac_saturation_single_bss"
    with source.open(newline="", encoding="utf-8") as stream:
        mac_rows = [row for row in csv.DictReader(stream) if int(row["bss"]) == 0]
    for flow_id, row in enumerate(mac_rows):
        row["flow_id"] = flow_id
    write_csv(mac_target / "flows.csv", mac_rows)
    shutil.copy2(base / "api1/topology.csv", mac_target / "topology.csv")
    queue_counts: dict[str, set[str]] = {}
    for row in mac_rows:
        queue_counts.setdefault(row["ac"], set()).add(row["source"])
    mac_manifest = {
        "name": "image_mac_saturation_single_bss",
        "classification": "single-BSS direct-MAC saturated four-AC baseline",
        "active_bss": 1,
        "sta_per_bss": 40,
        "interfering_bss": 0,
        "enabled_acs": ["VO", "VI", "BE", "BK"],
        "active_source_queues_by_ac": {ac: len(nodes) for ac, nodes in queue_counts.items()},
        "transport_stack": "none; direct WifiNetDevice injection",
        "payload_model": {
            "VO": "fixed 160 B", "VI": "clipped lognormal mean 1280 B sigma 0.35 [256,1500] B",
            "BE": "fixed 1500 B", "BK": "fixed 1600 B",
        },
        "full_buffer_enforcement": "each active EDCA queue refilled to configured backlog",
    }
    (mac_target / "manifest.json").write_text(
        json.dumps(mac_manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
