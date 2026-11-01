from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCENES = {"subset7": 7, "full19": 19}


def read(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class TgaxLoadVariantTest(unittest.TestCase):
    def test_catalog_labels_extensions(self):
        for scene in SCENES:
            root = ROOT / f"results/raw/tgax_s3_inputs_20261021/{scene}"
            catalog = json.loads((root / "load_variant_catalog.json").read_text(encoding="utf-8"))
            labels = {row["name"]: row["classification"] for row in catalog}
            self.assertEqual(labels["load100"], "official")
            for name in ("mix_min1", "mix_min2", "load025", "load050", "load075", "load125", "load150", "rt125", "rt150"):
                self.assertEqual(labels[name], "TGax-based extension scenario")
            self.assertEqual(labels["candidate_ftp_bk"], "candidate research configuration")
            self.assertEqual(labels["candidate_voip8_gaming4_ftp_bk"],
                             "TGax-model-driven custom concurrency candidate")

    def test_component_counts_per_bss(self):
        expected = {"candidate_ftp_bk": 10, "candidate_voip8_gaming4_ftp_bk": 30,
                    "mix_min1": 10, "mix_min2": 20,
                    "load025": 25, "load050": 50, "load075": 75, "load100": 100,
                    "load125": 125, "load150": 150, "rt125": 120, "rt150": 140}
        for scene, bss_count in SCENES.items():
            root = ROOT / f"results/raw/tgax_s3_inputs_20261021/{scene}"
            for name, per_bss in expected.items():
                rows = read(root / name / "flows.csv")
                self.assertEqual(len(rows), bss_count * per_bss)
                for bss in range(bss_count):
                    self.assertEqual(sum(int(row["bss"]) == bss for row in rows), per_bss)

    def test_realtime_variants_keep_be_counts(self):
        for scene in SCENES:
            root = ROOT / f"results/raw/tgax_s3_inputs_20261021/{scene}"
            official = Counter((row["service"], row["direction"])
                               for row in read(root / "load100/flows.csv") if row["bss"] == "0")
            for name in ("rt125", "rt150"):
                actual = Counter((row["service"], row["direction"])
                                 for row in read(root / name / "flows.csv") if row["bss"] == "0")
                for service in ("D3", "D4", "U2"):
                    for direction in ("DL", "UL"):
                        self.assertEqual(actual[service, direction], official[service, direction])

    def test_generator_parameters_are_not_overridden(self):
        for scene in SCENES:
            root = ROOT / f"results/raw/tgax_s3_inputs_20261021/{scene}"
            for manifest_path in root.glob("*/manifest.json"):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertFalse(manifest["generator_internals_changed"])

    def test_subset7_variants_are_nested_and_stream_stable(self):
        root = ROOT / "results/raw/tgax_s3_inputs_20261021/subset7"
        rows = {name: read(root / name / "flows.csv")
                for name in ("load025", "load050", "load075", "load100", "load125", "load150")}
        maps = {name: {row["stream_id"]: (row["bss"], row["service"], row["assignment"],
                                               row["direction"], row["sta"])
                       for row in values}
                for name, values in rows.items()}
        for lower, higher in zip(("load025", "load050", "load075", "load100", "load125"),
                                 ("load050", "load075", "load100", "load125", "load150")):
            self.assertTrue(set(maps[lower]).issubset(maps[higher]))
            for stream_id, logical_flow in maps[lower].items():
                self.assertEqual(logical_flow, maps[higher][stream_id])

    def test_candidate_only_moves_ftp_to_bk(self):
        root = ROOT / "results/raw/tgax_s3_inputs_20261021/subset7"
        candidate = read(root / "candidate_ftp_bk/flows.csv")
        self.assertTrue(candidate)
        for row in candidate:
            if row["service"] in {"D3", "U2"}:
                self.assertEqual(row["ac"], "BK")
            else:
                expected = {"D1": "VI", "D2": "VI", "D4": "BE", "D5": "VI",
                            "D6": "VO", "U1": "VI", "U3": "VI"}[row["service"]]
                self.assertEqual(row["ac"], expected)

    def test_realtime_pressure_candidate_counts_and_stability(self):
        root = ROOT / "results/raw/tgax_s3_inputs_20261021/subset7"
        old = read(root / "candidate_ftp_bk/flows.csv")
        new = read(root / "candidate_voip8_gaming4_ftp_bk/flows.csv")
        for bss in range(7):
            actual = Counter((r["service"], r["direction"]) for r in new if int(r["bss"]) == bss)
            self.assertEqual(actual["D6", "DL"], 8)
            self.assertEqual(actual["D6", "UL"], 8)
            self.assertEqual(actual["D5", "DL"], 4)
            self.assertEqual(actual["U3", "UL"], 4)
            for key in (("D1", "DL"), ("D2", "DL"), ("D3", "DL"), ("D4", "DL"),
                        ("U1", "UL"), ("U2", "UL")):
                self.assertEqual(actual[key], 1)
            target = [r for r in new if int(r["bss"]) == bss]
            active = {r["sink"] if r["direction"] == "DL" else r["source"] for r in target}
            old_target = [r for r in old if int(r["bss"]) == bss]
            old_active = {r["sink"] if r["direction"] == "DL" else r["source"]
                          for r in old_target}
            # Seven added VoIP pairs and three added flows in each Gaming
            # direction use thirteen previously idle STAs in every BSS.
            self.assertEqual(len(active), len(old_active) + 13)
        old_map = {(r["bss"], r["service"], r["direction"]): (r["sta"], r["stream_id"])
                   for r in old}
        new_first = {(r["bss"], r["service"], r["direction"]): (r["sta"], r["stream_id"])
                     for r in new if r["assignment"] == "0"}
        self.assertEqual(old_map, new_first)
        self.assertTrue(all(r["ac"] == "BK" for r in new if r["service"] in {"D3", "U2"}))


if __name__ == "__main__":
    unittest.main()
