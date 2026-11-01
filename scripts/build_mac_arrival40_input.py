"""Build the independent one-BSS, direct-MAC TGax-arrival input set."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT


SOURCE_TOPOLOGY = ROOT / "results/raw/tgax_mac_fullbuffer40_inputs_20261028/topology.csv"
SOURCE_FLOWS = ROOT / "results/raw/tgax_s3_inputs_20261021/subset7/candidate_full40_multiac_v6/flows.csv"
OUT = ROOT / "results/raw/tgax_mac_arrival40_inputs_20261029"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    # ns-3's deliberately small CSV reader does not strip an UTF-8 BOM from
    # the first header field, so write plain UTF-8.
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    topology = read(SOURCE_TOPOLOGY)
    flows = [row for row in read(SOURCE_FLOWS) if int(row["bss"]) == 0]
    for row in flows:
        row["reference_transport"] = row["transport"]
        row["transport"] = "NONE"
    topology_fields = list(topology[0])
    flow_fields = list(flows[0])
    if "reference_transport" not in flow_fields:
        flow_fields.append("reference_transport")
    write(OUT / "topology.csv", topology, topology_fields)
    write(OUT / "flows.csv", flows, flow_fields)

    counts: dict[str, int] = {}
    for row in flows:
        key = f'{row["model"]}/{row["direction"]}/{row["ac"]}'
        counts[key] = counts.get(key, 0) + 1
    manifest = {
        "identity": "single_bss_mac_arrival40_default",
        "topology_source": str(SOURCE_TOPOLOGY.relative_to(ROOT)).replace("\\", "/"),
        "traffic_assignment_source": str(SOURCE_FLOWS.relative_to(ROOT)).replace("\\", "/"),
        "ap_count": sum(row["role"] == "AP" for row in topology),
        "sta_count": sum(row["role"] == "STA" for row in topology),
        "configured_flow_rows": len(flows),
        "runtime_http_request_rows_added": sum(row["model"] == "HTTP" for row in flows),
        "transport_runtime": "NONE; direct WifiMac enqueue",
        "reference_transport_retained_as_metadata": True,
        "counts": counts,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
