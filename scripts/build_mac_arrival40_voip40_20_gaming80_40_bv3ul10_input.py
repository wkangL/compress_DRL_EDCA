"""Build the independent one-BSS workload requested for the next CMDP run.

Only the requested concurrency counts differ from the preceding high-Gaming
scenario.  Packet-size and arrival-state models remain implemented by the
existing ns-3 direct-MAC generators; this file only assigns model instances to
STAs and unique random streams.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT


BASE = ROOT / "results/raw/tgax_mac_arrival40_gaming160_inputs_20261029"
OUT = ROOT / "results/raw/tgax_mac_arrival40_voip40_20_gaming80_40_bv3ul10_inputs_20261030"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clone_flow(template: dict[str, str], *, flow_id: int, stream_id: int,
               sta: dict[str, str], service: str, assignment: int) -> dict[str, str]:
    row = dict(template)
    row.update({
        "flow_id": str(flow_id),
        "stream_id": str(stream_id),
        "service": service,
        "assignment": str(assignment),
        "sta": sta["sta"],
        "source": "ap0" if row["direction"] == "DL" else sta["node_label"],
        "sink": sta["node_label"] if row["direction"] == "DL" else "ap0",
    })
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    topology = read(BASE / "topology.csv")
    base = read(BASE / "flows.csv")
    stas = sorted((row for row in topology if row["role"] == "STA"),
                  key=lambda row: int(row["sta"]))
    if len(stas) != 40:
        raise RuntimeError(f"Expected 40 STAs, found {len(stas)}")

    # Preserve every unmentioned row and the preceding 80-flow Gaming UL set.
    # Of the two prior DL Gaming replicas per STA, keep R0 only (40 flows).
    rows = [
        row for row in base
        if not (row["model"] == "GMG" and row["direction"] == "DL"
                and row["service"].endswith("_R1"))
    ]

    next_id = 300000
    next_stream = 300000

    # The previous scenario already has one DL VoIP flow.  Rebuild all 20 DL
    # assignments explicitly so the requested count and STA coverage are clear.
    voip_dl_template = next(row for row in rows if row["model"] == "VOIP" and row["direction"] == "DL")
    rows = [row for row in rows if not (row["model"] == "VOIP" and row["direction"] == "DL")]
    for index, sta in enumerate(stas[:20]):
        rows.append(clone_flow(voip_dl_template, flow_id=next_id, stream_id=next_stream,
                               sta=sta, service=f"D6_VOIP20_R{index}", assignment=index))
        next_id += 1
        next_stream += 1

    # Retain the original BV3 UL instance and add nine independent instances on
    # distinct STAs.  Rebuilding all ten makes the assignment deterministic.
    bv3_ul_template = next(row for row in rows if row["model"] == "BV3" and row["direction"] == "UL")
    rows = [row for row in rows if not (row["model"] == "BV3" and row["direction"] == "UL")]
    bv3_sta_indices = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38]
    for index, sta_index in enumerate(bv3_sta_indices):
        sta = stas[sta_index]
        rows.append(clone_flow(bv3_ul_template, flow_id=next_id, stream_id=next_stream,
                               sta=sta, service=f"U1_BV3_10_R{index}", assignment=index))
        next_id += 1
        next_stream += 1

    rows.sort(key=lambda row: int(row["flow_id"]))
    write(OUT / "topology.csv", topology, list(topology[0]))
    write(OUT / "flows.csv", rows, list(base[0]))

    counts: dict[str, int] = {}
    for row in rows:
        key = f'{row["model"]}/{row["direction"]}/{row["ac"]}'
        counts[key] = counts.get(key, 0) + 1
    expected = {
        "VOIP/UL/VO": 40, "VOIP/DL/VO": 20,
        "GMG/UL/VI": 80, "GMG/DL/VI": 40,
        "BV3/UL/VI": 10,
        "HTTP/DL/BE": 20, "FTP/UL/BE": 12,
        "HTTP/DL/BK": 16, "FTP/UL/BK": 8,
    }
    if counts != expected:
        raise RuntimeError(f"Flow-count mismatch: {counts!r} != {expected!r}")
    manifest = {
        "identity": "single_bss_mac_arrival40_voip40_20_gaming80_40_bv3ul10",
        "base": str(BASE.relative_to(ROOT)).replace("\\", "/"),
        "transport_runtime": "NONE; direct native WifiMacQueue admission; no custom watermark",
        "arrival_models": "unchanged TGax-based model implementations; only concurrency counts changed",
        "configured_flow_rows": len(rows),
        "counts": counts,
        "unchanged_from_previous": {
            "gaming_ul": 80, "http_be_dl": 20, "ftp_be_ul": 12,
            "http_bk_dl": 16, "ftp_bk_ul": 8,
        },
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
