import unittest
import numpy as np
from reporting.sta_state import StaHistory,StateScaler
from reporting.report_cache import ReportCache,Report
from reporting.report_generator import ReportGenerator
from sim.backend.base_backend import Event


class ReportingTests(unittest.TestCase):
    def test_generated_history_not_current_state(self) -> None:
        h=StaHistory(1,10); c=ReportCache(1,10,0); g=ReportGenerator(h,c)
        events=[Event("S",(.1,1,0,123,5,2,0,0,1,0,0,0,0)),Event("G",(.11,0,111)),
                Event("S",(.2,1,0,999,0,0,0,0,0,0,0,0,0)),Event("R",(.21,0,.11,111))]
        g.process(events)
        x,age,valid=c.snapshot(.3)
        self.assertEqual(x[0,0,-1,0],123)
        self.assertAlmostEqual(float(age[0]),.19,places=6)
        self.assertEqual(valid[0],1)
        self.assertEqual(h.snapshot(0)[0,-1,0],999)

    def test_missing_reports_and_version_rejection(self) -> None:
        c=ReportCache(3,10,3)
        self.assertEqual(c.snapshot(0)[2].sum(),0)
        with self.assertRaises(ValueError): c.receive(Report(0,0,.1,1,2,np.zeros((4,10,6))))
        c.receive(Report(0,0,.1,1,3,np.zeros((4,10,6))))
        c.sync_version(4)
        self.assertEqual(c.snapshot(1)[2].sum(),0)

    def test_scaler_evaluation_leakage(self) -> None:
        x=np.ones((10,6),dtype=np.float32)
        s=StateScaler(100)
        with self.assertRaises(ValueError): s.fit(x,"eval")
        s.fit(x)
        before=s.mean.copy(); s.transform(x*2)
        np.testing.assert_array_equal(before,s.mean)
        with self.assertRaises(ValueError): s.fit(x)


if __name__=="__main__": unittest.main()
