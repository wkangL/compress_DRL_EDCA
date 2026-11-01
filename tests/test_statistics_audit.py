import csv
import tempfile
import unittest
from pathlib import Path

from scripts.audit_statistics import analyze


HEADER = ("time_s", "node", "event", "ac", "value", "uid", "sta")


class StrictStatisticsAuditTests(unittest.TestCase):
    def write_trace(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream); writer.writerow(HEADER); writer.writerows(rows)

    def test_window_boundary_enqueue_and_terminal_conservation(self):
        rows = [
            (9.9, 0, "OFFERED", 0, 160, 1, 3),
            (9.9, 0, "ENQUEUED", 0, 160, 1, 3),
            (10.01, 0, "ACKED", 0, 110, 1, 3),
            (10.1, 4, "OFFERED", 1, 1200, 2, 3),
            (10.1, 4, "ENQUEUED", 1, 1200, 2, 3),
            (10.2, 4, "DATA_ATTEMPT", 1, 0, 2, 3),
            (10.3, 4, "DROP", 1, 2, 2, 3),
            (10.2, 4, "OFFERED", 1, 1200, 3, 3),
            (10.2, 4, "DROP", 1, 0, 3, 3),
            (10.3, 4, "OFFERED", 1, 1200, 4, 3),
            (10.3, 4, "ENQUEUED", 1, 1200, 4, 3),
            (70, 4, "QUEUE_PACKETS", 1, 1, 0, -1),
        ]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); trace = root / "trace.csv"; self.write_trace(trace, rows)
            report = analyze(trace, 10, 70, root / "out")
            self.assertTrue(report["conservation_passed"])
            with (root / "out/accounting_by_sta_ac_direction.csv").open() as stream:
                data = list(csv.DictReader(stream))
            ul_vi = next(r for r in data if r["direction"] == "UL")
            self.assertEqual((int(ul_vi["generated"]), int(ul_vi["accepted"]),
                              int(ul_vi["queue_rejected"]), int(ul_vi["retry_limit"]),
                              int(ul_vi["unfinished_end"])), (3, 2, 1, 1, 1))

    def test_duplicate_terminal_is_rejected(self):
        rows = [(10.1, 1, "OFFERED", 0, 160, 1, 0),
                (10.1, 1, "ENQUEUED", 0, 160, 1, 0),
                (10.2, 1, "ACKED", 0, 100, 1, 0),
                (10.3, 1, "DROP", 0, 2, 1, 0)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); trace = root / "trace.csv"; self.write_trace(trace, rows)
            with self.assertRaisesRegex(AssertionError, "multiple terminal"):
                analyze(trace, 10, 70, root / "out")

    def test_missing_enqueue_decision_is_rejected(self):
        rows = [(10.1, 1, "OFFERED", 0, 160, 1, 0)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); trace = root / "trace.csv"; self.write_trace(trace, rows)
            with self.assertRaisesRegex(AssertionError, "enqueue decision"):
                analyze(trace, 10, 70, root / "out")


if __name__ == "__main__":
    unittest.main()
