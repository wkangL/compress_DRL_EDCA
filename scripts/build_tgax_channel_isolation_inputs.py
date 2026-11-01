"""Build paired two-BSS co-channel/orthogonal-channel validation inputs.

These are explicit mechanism-test variants, not official Scenario 3 performance
configurations.  Coordinates and the selected BV6 flow are copied unchanged
from the deterministic Scenario 3 input.
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path: Path, data: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)


def main() -> None:
    source = ROOT / "results/raw/tgax_s3_inputs_20261021/subset7"
    out = ROOT / "results/raw/tgax_s3_channel_isolation_inputs_20261021"
    topology = [r for r in read(source / "topology.csv") if int(r["bss"]) in {0, 1}]
    # One official BV6 source in BSS 1; BSS 0 is a sensing observer apart from
    # native management traffic.  The only difference in the pair is channel.
    flows = [r for r in read(source / "flows.csv")
             if int(r["bss"]) == 1 and r["service"] == "D1"][:1]
    for name, second_channel in (("cochannel", "42"), ("orthogonal", "58")):
        variant = [dict(r) for r in topology]
        for r in variant:
            if int(r["bss"]) == 1:
                r["channel"] = second_channel
        write(out / name / "topology.csv", variant)
        write(out / name / "flows.csv", flows)
        (out / name / "README.txt").write_text(
            "Mechanism-test variant only. Same Scenario-3-derived coordinates; "
            f"BSS0=channel42, BSS1=channel{second_channel}; one BSS1 BV6 flow.\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
