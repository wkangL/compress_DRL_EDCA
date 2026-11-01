"""Build the independent single-BSS 40-STA direct-MAC full-buffer input."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT


SOURCE = ROOT / "results/raw/tgax_s3_inputs_20261021/subset7/topology.csv"
OUTPUT = ROOT / "results/raw/tgax_mac_fullbuffer40_inputs_20261028"
FIELDS = ("flow_id", "stream_id", "bss", "service", "model", "assignment",
          "direction", "transport", "ac", "sta", "source", "sink")


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        topology = [row for row in csv.DictReader(stream) if int(row["bss"]) == 0]
    assert len(topology) == 41
    assert sum(row["role"] == "STA" for row in topology) == 40

    rows: list[dict] = []
    assignments = {
        "VO": range(0, 20),
        "VI": range(20, 40),
        # Balance extra low-priority queues across the two primary groups.
        "BE": (*range(0, 10), *range(20, 30)),
        "BK": (*range(10, 16), *range(30, 36)),
    }
    model = {"VO": "VOIP_SIZE", "VI": "GAMING_SIZE",
             "BE": "GENERIC_DATA_SIZE", "BK": "BACKGROUND_DATA_SIZE"}
    for ac, stations in assignments.items():
        for sta in stations:
            index = len(rows)
            rows.append({
                "flow_id": index, "stream_id": index, "bss": 0,
                "service": f"MAC_FB_{ac}", "model": model[ac],
                "assignment": sta, "direction": "UL", "transport": "NONE",
                "ac": ac, "sta": sta, "source": f"b0_sta{sta}", "sink": "ap0",
            })
    assert len(rows) == 72
    assert {row["source"] for row in rows if row["ac"] in {"VO", "VI"}} == {
        f"b0_sta{sta}" for sta in range(40)
    }

    write_csv(OUTPUT / "topology.csv", topology, list(topology[0]))
    write_csv(OUTPUT / "fullbuffer_sources.csv", rows, FIELDS)
    manifest = {
        "name": "single_bss_mac_fullbuffer40_default",
        "classification": "TGax-geometry-based independent MAC full-buffer research scenario",
        "topology_source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "bss": 1, "ap": 1, "sta": 40, "channel": 42,
        "primary_high_priority": {"VO_UL": 20, "VI_UL": 20},
        "additional": {"BE_UL": 20, "BK_UL": 12},
        "active_source_queues": 72,
        "transport_stack": "none: no IPv4/TCP/UDP/Socket objects in traffic mode",
        "injection": "WifiMac::Enqueue(packet, AP, STA, TID) with LLC local protocol header",
        "queue_control": {"low_packets": 32, "high_packets": 64, "poll_us": 100},
        "payload_bytes": {
            "VO": "33 fixed (TGax active VoIP payload size)",
            "VI": "TGax uplink Gaming largest-extreme-value mean=158 scale=26.2, clipped [64,1500]",
            "BE": "discrete uniform [800,1500]",
            "BK": "discrete uniform [1200,1500]",
        },
        "arrival_process": "controlled full buffer; original application inter-arrival/state/read-time models disabled",
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
