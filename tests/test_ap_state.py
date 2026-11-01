import unittest

import numpy as np

from project_config import ROOT, load_config
from sim.backend.base_backend import Event
from sim.statistics.ap_state import GLOBAL_DIM, GLOBAL_FEATURES, ap_features


class ApStateTests(unittest.TestCase):
    def test_ten_features_are_ap_local_and_single_window(self):
        events = [
            Event("B", (0.0, 0.4, 1)),
            Event("B", (0.2, 0.5, 2)),  # overlap: total busy time is 0.7 s
            Event("D", (0.5, 2, 0, 1_000_000, 1.0, 3)),  # UL VO: visible at AP receiver
            Event("D", (0.6, 4, 1, 2_000_000, 1.0, 0)),  # DL VI: intentionally excluded
        ]
        x = ap_features(events, 0.0, 1.0, 16.0, np.array([0, 999, 1_000_000, 9]),
                        associated_sta=10, associated_scale=64, queue_byte_scale=1_000_000)
        self.assertEqual(x.shape, (GLOBAL_DIM,))
        self.assertEqual(len(GLOBAL_FEATURES), GLOBAL_DIM)
        self.assertAlmostEqual(float(x[0]), 0.7, places=6)
        np.testing.assert_allclose(x[1:5], [0.5, 0, 0, 0], atol=1e-7)
        self.assertAlmostEqual(float(x[-1]), 10 / 64, places=7)

    def test_config_declares_exact_state_schema(self):
        config = load_config(ROOT / "configs/full.yaml")
        self.assertEqual(tuple(config["state"]["global_features"]), GLOBAL_FEATURES)
        self.assertEqual(config["schema_version"], 2)

    def test_invalid_scales_and_time_rejected(self):
        for args in ((1, 0, 27, 10, 64, 1e6), (0, 1, 0, 10, 64, 1e6),
                     (0, 1, 27, 10, 0, 1e6), (0, 1, 27, 10, 64, 0)):
            start, end, capacity, associated, associated_scale, queue_scale = args
            with self.assertRaises(ValueError):
                ap_features([], start, end, capacity, np.zeros(4), associated,
                            associated_scale, queue_scale)


if __name__ == "__main__":
    unittest.main()
