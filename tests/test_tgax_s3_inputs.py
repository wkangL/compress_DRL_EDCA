from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "results/raw/tgax_s3_inputs_20261021"


def read(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TgaxScenario3InputTest(unittest.TestCase):
    def test_scale_is_explicit(self):
        expected = {"main3": (3, 30), "api1": (1, 40), "subset7": (7, 280), "full19": (19, 760)}
        for variant, (aps, stas) in expected.items():
            data = read(INPUTS / variant / "topology.csv")
            self.assertEqual(sum(r["role"] == "AP" for r in data), aps)
            self.assertEqual(sum(r["role"] == "STA" for r in data), stas)

    def test_official_service_count_per_bss(self):
        expected = Counter({("D1", "DL"): 12, ("D2", "DL"): 8,
                            ("D3", "DL"): 4, ("D4", "DL"): 12,
                            ("D5", "DL"): 16, ("D6", "DL"): 12,
                            ("D6", "UL"): 12, ("U1", "UL"): 4,
                            ("U2", "UL"): 4, ("U3", "UL"): 16})
        flows = read(INPUTS / "full19" / "flows.csv")
        for bss in range(19):
            actual = Counter((r["service"], r["direction"])
                             for r in flows if int(r["bss"]) == bss)
            self.assertEqual(actual, expected)

    def test_main3_uses_one_quarter_service_counts(self):
        expected = Counter({("D1", "DL"): 3, ("D2", "DL"): 2,
                            ("D3", "DL"): 1, ("D4", "DL"): 3,
                            ("D5", "DL"): 4, ("D6", "DL"): 3,
                            ("D6", "UL"): 3, ("U1", "UL"): 1,
                            ("U2", "UL"): 1, ("U3", "UL"): 4})
        flows = read(INPUTS / "main3" / "flows.csv")
        for bss in range(3):
            actual = Counter((r["service"], r["direction"])
                             for r in flows if int(r["bss"]) == bss)
            self.assertEqual(actual, expected)

    def test_main3_ap_triangle_has_30m_sides(self):
        aps = [r for r in read(INPUTS / "main3" / "topology.csv") if r["role"] == "AP"]
        points = [(float(r["x"]), float(r["y"])) for r in aps]
        distances = [math.dist(a, b) for i, a in enumerate(points) for b in points[i + 1:]]
        self.assertEqual(len(distances), 3)
        for value in distances:
            self.assertAlmostEqual(value, 30.0, places=9)

    def test_reuse3_nearest_ap_distance_is_30m(self):
        aps = [r for r in read(INPUTS / "full19" / "topology.csv") if r["role"] == "AP"]
        points = [(float(r["x"]), float(r["y"])) for r in aps]
        nearest = min(math.dist(a, b) for i, a in enumerate(points) for b in points[i + 1:])
        self.assertAlmostEqual(nearest, 30.0, places=9)

    def test_every_retained_sta_matches_strongest_reference_ap(self):
        refs = {r["ref_id"]: (float(r["x"]), float(r["y"]))
                for r in read(INPUTS / "full19" / "reference_61_cells.csv")}
        aps = {r["strongest_ref"]: (float(r["x"]), float(r["y"]))
               for r in read(INPUTS / "full19" / "topology.csv") if r["role"] == "AP"}
        self.assertEqual(set(aps), set(r["strongest_ref"] for r in read(INPUTS / "full19" / "topology.csv") if r["role"] == "AP"))
        for row in read(INPUTS / "full19" / "topology.csv"):
            if row["role"] != "STA":
                continue
            self.assertIn(row["strongest_ref"], aps)
            self.assertGreaterEqual(float(row["strongest_margin_db"]), 0.0)
            self.assertEqual(aps[row["strongest_ref"]], refs[row["strongest_ref"]])


if __name__ == "__main__":
    unittest.main()
