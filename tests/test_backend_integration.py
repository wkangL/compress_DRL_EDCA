"""Small real-ns-3 integration checks, explicitly opt in with WIFI_INTEGRATION=1."""
import os,tempfile,unittest
from pathlib import Path
import numpy as np
from project_config import ROOT,load_config
from rl.environment import ControlEnvironment


@unittest.skipUnless(os.environ.get("WIFI_INTEGRATION")=="1","real ns-3 integration is opt-in")
class BackendTests(unittest.TestCase):
    def test_causal_reports_and_interval_statistics(self) -> None:
        c=load_config(ROOT/"configs/quick.yaml"); c["network"]["num_sta"]=1
        scenario={"offered_mbps":1.,"ac_mix":[0,0,1,0],"ul_fraction":1.}
        with tempfile.TemporaryDirectory(dir=ROOT/"results/raw") as tmp:
            env=ControlEnvironment(c,"compressed_async_age",scenario,123,Path(tmp),27.,0)
            try:
                obs,r,m=env.step(env.action,1.)
                self.assertEqual(obs["history"].shape,(1,4,10,6))
                self.assertGreater(obs["valid"].sum(),0)
                self.assertTrue(np.isfinite(obs["global_stats"]).all())
                self.assertEqual(obs["global_stats"].shape,(10,))
                self.assertEqual(m["collision_rate"],0)
                self.assertEqual(m["drop_rate"],0)
                self.assertLess(obs["report_time"][0],obs["decision_time"])
                self.assertGreater(m["report_rate_per_sta"],0)
                before=obs["history"].copy()
                next_obs,_,_=env.step(env.action,1.)
                np.testing.assert_array_equal(before,obs["history"])
                self.assertGreater(next_obs["decision_time"],obs["decision_time"])
            finally: env.close()


if __name__=="__main__": unittest.main()
