import random
import unittest
from copy import deepcopy
import numpy as np
import torch
from project_config import ROOT,load_config
from reporting.comparison import empirical_ccdf,balanced_ccdf,relative_change
from rl.monitoring import evaluation_stages,isolated_random_state


class ComparisonMonitoringTests(unittest.TestCase):
    def test_ccdf_ties_and_strict_exceedance(self):
        np.testing.assert_allclose(empirical_ccdf(np.array([1,1,2,4]),np.array([0,1,2,4])),[1,.5,.25,0])
        self.assertTrue(np.isnan(empirical_ccdf(np.array([]),np.array([0]))).all())

    def test_ccdf_weights_seeds_not_packet_counts(self):
        values={1:{101:np.zeros(1000),102:np.ones(1)*10},2:{101:np.ones(1)*10,102:np.ones(10)*10}}
        np.testing.assert_allclose(balanced_ccdf(values,np.array([5.])),[.75])

    def test_percentages_preserve_regression_and_zero(self):
        self.assertAlmostEqual(relative_change(10,12),20)
        self.assertAlmostEqual(relative_change(10,8),-20)
        self.assertIsNone(relative_change(0,1))
        self.assertIsNone(relative_change(0,0))

    def test_monitor_schedule_includes_learning_and_target_sync(self):
        config=load_config(ROOT/"configs/full.yaml")
        options=dict(every_stages=10,duration_s=30,evaluation_seeds=[201,202])
        self.assertEqual(evaluation_stages(config,options,1),[1,42,50,60,70,80,90,100])
        hard=deepcopy(config); hard["training"]["target_update_mode"]="hard"
        self.assertEqual(evaluation_stages(hard,options,1),[1,42,44,50,60,70,80,90,100])

    def test_monitoring_restores_all_global_random_streams_on_error(self):
        random.seed(6); np.random.seed(6); torch.manual_seed(6)
        expected=(random.random(),np.random.rand(),torch.rand(3))
        random.seed(6); np.random.seed(6); torch.manual_seed(6)
        with self.assertRaises(RuntimeError):
            with isolated_random_state():
                random.seed(99); np.random.seed(99); torch.manual_seed(99)
                raise RuntimeError("evaluation interrupted")
        self.assertEqual(random.random(),expected[0]); self.assertEqual(np.random.rand(),expected[1])
        self.assertTrue(torch.equal(torch.rand(3),expected[2]))


if __name__=="__main__": unittest.main()
