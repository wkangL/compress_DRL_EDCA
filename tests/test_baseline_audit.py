import csv
import tempfile
import unittest
from pathlib import Path
from scripts.audit_baseline import analyze_trace


class AccountingTests(unittest.TestCase):
    def test_losses_and_censoring_do_not_disappear_from_deadline_failure(self):
        # One timely success, one overflow, one expired and two pending;
        # the last pending packet lacks enough observation time for the deadline.
        rows=[(0,0,'OFFERED',0,160,1),(.005,0,'ACKED',0,5,1),
              (.1,0,'OFFERED',0,160,2),(.1,0,'UNREPORTED_ENQUEUE_DROP',0,0,2),
              (.2,0,'OFFERED',0,160,3),(.4,0,'DROP',0,1,3),
              (.5,0,'OFFERED',0,160,4),(.995,0,'OFFERED',0,160,5),
              (1,0,'QUEUE_PACKETS',0,2,0),(1,0,'QUEUE_LIMIT',0,100,0)]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'results') as d:
            p=Path(d)/'trace.csv'
            with p.open('w',newline='') as f:
                w=csv.writer(f); w.writerow(['time_s','node','event','ac','value','uid']); w.writerows(rows)
            r=analyze_trace(p,1,Path(d)/'out')['rows'][0]
            self.assertEqual(r['success_ccdf_20ms'],0)
            self.assertEqual(r['eligible_20ms'],4)
            self.assertEqual(r['not_delivered_by_20ms_fraction'],.75)
            self.assertEqual(r['pending'],2)
            self.assertEqual(r['true_drop_fraction'],.4)
            with p.open('a',newline='') as f: csv.writer(f).writerow((1,0,'QUEUE_PACKETS',0,1,0))
            with self.assertRaisesRegex(ValueError,'Queue conservation'):
                analyze_trace(p,1,Path(d)/'bad')


if __name__=='__main__': unittest.main()
