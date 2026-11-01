"""Load and audit the unchanged 16 profiles recovered from the legacy project."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ACS = ("VO", "VI", "BE", "BK")


def load_profiles(path: str | Path) -> list[dict]:
    profiles = json.loads(Path(path).read_text(encoding="utf-8"))
    for profile in profiles:
        if set(profile) != set(ACS):
            raise ValueError("Each profile must specify all four ACs")
        for ac in ACS:
            p = profile[ac]
            for name in ("CWmin", "CWmax"):
                cw = p[name]
                if not isinstance(cw, int) or cw < 0 or cw > 32767 or cw & (cw + 1):
                    raise ValueError(f"{name} must be 2^m-1")
            if p["CWmin"] > p["CWmax"] or not 2 <= p["AIFSN"] <= 15:
                raise ValueError("Illegal EDCA bounds")
            units = p["TXOP_ms"] * 1000 / 32
            if abs(units - round(units)) > 1e-8 or units < 0:
                raise ValueError("TXOP must be a nonnegative multiple of 32 us")
    return profiles


def export_profiles(profiles: list[dict], directory: Path) -> str:
    """Write all 64 AC rows as CSV and return a Markdown table."""
    directory.mkdir(parents=True, exist_ok=True)
    columns = ["profile", "AC", "AIFSN", "CWmin", "CWmax", "TXOP_ms"]
    rows = [[i, ac, *(p[ac][key] for key in columns[2:])]
            for i, p in enumerate(profiles) for ac in ACS]
    with (directory / "edca_profiles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    return "\n".join(["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])
