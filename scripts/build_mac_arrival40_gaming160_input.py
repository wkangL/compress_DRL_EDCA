"""Expand the one-BSS direct-MAC arrival scenario to 160 Gaming flows."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT


BASE = ROOT / "results/raw/tgax_mac_arrival40_inputs_20261029"
OUT = ROOT / "results/raw/tgax_mac_arrival40_gaming160_inputs_20261029"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    topology = read(BASE / "topology.csv")
    base = read(BASE / "flows.csv")
    non_gaming = [row for row in base if row["model"] != "GMG"]
    sta_labels = sorted((row for row in topology if row["role"] == "STA"), key=lambda row: int(row["sta"]))
    gaming: list[dict[str, str]] = []
    next_id = 200000
    next_stream = 200000
    for direction in ("DL", "UL"):
        for replica in range(2):
            for sta in sta_labels:
                downlink = direction == "DL"
                gaming.append({
                    "flow_id": str(next_id), "stream_id": str(next_stream), "bss": "0",
                    "service": f'{"D5" if downlink else "U3"}_GAME2_R{replica}',
                    "model": "GMG", "assignment": sta["sta"], "direction": direction,
                    "transport": "NONE", "ac": "VI", "sta": sta["sta"],
                    "source": "ap0" if downlink else sta["node_label"],
                    "sink": sta["node_label"] if downlink else "ap0",
                    "reference_transport": "UDP",
                })
                next_id += 1; next_stream += 1
    flows = non_gaming + gaming
    fields = list(base[0])
    write(OUT / "topology.csv", topology, list(topology[0]))
    write(OUT / "flows.csv", flows, fields)
    counts: dict[str, int] = {}
    for row in flows:
        key = f'{row["model"]}/{row["direction"]}/{row["ac"]}'
        counts[key] = counts.get(key, 0) + 1
    manifest = {
        "identity": "single_bss_mac_arrival40_gaming160_default",
        "base": str(BASE.relative_to(ROOT)).replace("\\", "/"),
        "only_workload_change": "replace one DL Gaming flow with two DL and two UL Gaming flows per STA",
        "gaming_dl_flows": 80, "gaming_ul_flows": 80, "gaming_total_flows": 160,
        "configured_flow_rows": len(flows),
        "runtime_http_request_rows_added": sum(row["model"] == "HTTP" for row in flows),
        "transport_runtime": "NONE; direct WifiMac enqueue",
        "counts": counts,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
