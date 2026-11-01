"""Recovery must preserve the saved boundary and reproduce uninterrupted training."""
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from project_config import ROOT,load_config
from rl.artifacts import atomic_output
from rl.trainer import resume_logs,train
from scripts.audit_run import audit_run
from rl.training_budget import training_budget


class RecoveryTests(unittest.TestCase):
    def test_full_budget_reaches_epsilon_end(self):
        config=load_config(ROOT/"configs/full.yaml")
        budget=training_budget(config)
        self.assertEqual(budget["total_steps"],12000)
        self.assertEqual(budget["first_update_stage"],42)
        self.assertEqual(budget["final_action_epsilon"],.05)
        config["training"]["epsilon_steps"]=100000
        with self.assertRaisesRegex(ValueError,"before epsilon decay"): training_budget(config)

    def test_failed_publish_preserves_previous_artifact(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"results/raw") as tmp:
            path=Path(tmp)/"checkpoint.pt"
            path.write_bytes(b"previous")
            with self.assertRaises(OSError),patch("rl.artifacts.os.replace",side_effect=OSError("interrupted")):
                with atomic_output(path,binary=True) as stream: stream.write(b"incomplete")
            self.assertEqual(path.read_bytes(),b"previous")
            self.assertEqual(list(Path(tmp).glob("*.tmp")),[])

    def test_legacy_resume_filters_uncommitted_logs_and_rejects_missing(self):
        saved=dict(stage=0,steps=1,updates=0,config={"training":{"episodes_per_stage":1}})
        with tempfile.TemporaryDirectory(dir=ROOT/"results/raw") as tmp:
            path=Path(tmp)
            for name in ("steps.csv","episodes.csv"):
                (path/name).write_text("stage,value\n0,committed\n1,uncommitted\n",encoding="utf-8")
            steps,episodes,losses=resume_logs(saved,path)
            self.assertEqual(len(steps),1); self.assertEqual(len(episodes),1); self.assertEqual(losses,[])
            (path/"steps.csv").unlink()
            with self.assertRaisesRegex(ValueError,"missing or incomplete"): resume_logs(saved,path)

    def test_audit_rejects_missing_seed_even_with_complete_status(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"results/raw") as tmp:
            path=Path(tmp); config=load_config(ROOT/"configs/quick.yaml")
            (path/"config.json").write_text(json.dumps(config),encoding="utf-8")
            (path/"status.json").write_text(json.dumps(dict(state="complete",jobs=[
                dict(method="default",seed=1,status="complete",rows=8)])),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"complete configured"):
                audit_run(path,require_full_matrix=True)

    def assert_nested_equal(self,left,right):
        if isinstance(left,torch.Tensor): self.assertTrue(torch.equal(left,right))
        elif isinstance(left,np.ndarray): np.testing.assert_array_equal(left,right)
        elif isinstance(left,dict):
            self.assertEqual(left.keys(),right.keys())
            for key in left: self.assert_nested_equal(left[key],right[key])
        elif isinstance(left,(list,tuple)):
            self.assertEqual(len(left),len(right))
            for a,b in zip(left,right): self.assert_nested_equal(a,b)
        elif hasattr(left,"__dataclass_fields__"):
            self.assert_nested_equal(vars(left),vars(right))
        else: self.assertEqual(left,right)

    @unittest.skipUnless(os.environ.get("WIFI_INTEGRATION")=="1","real ns-3 integration is opt-in")
    def test_real_backend_exact_stage_resume(self):
        config=load_config(ROOT/"configs/quick.yaml")
        config["network"]["num_sta"]=1
        config["training"].update(stages=2,episodes_per_stage=1,episode_s=2.,batch_size=2,
                                  warmup=2,updates_per_stage=2,target_update_steps=2)
        with tempfile.TemporaryDirectory(dir=ROOT/"results/raw") as tmp:
            path=Path(tmp)
            uninterrupted=train(config,"compressed_async_age",7,path/"continuous")
            boundary=train(config,"compressed_async_age",7,path/"boundary",stop_after_stage=1)
            self.assertEqual(json.loads((boundary.parent/"status.json").read_text())["state"],"paused")
            # Simulate log writes after the saved stage and loss of adjacent CSVs.
            (boundary.parent/"steps.csv").write_text("corrupted after checkpoint",encoding="utf-8")
            (boundary.parent/"episodes.csv").unlink()
            from rl.evaluator import evaluate
            from rl.monitoring import isolated_random_state
            def monitor(checkpoint,saved):
                monitoring=deepcopy(config)
                monitoring["evaluation"].update(seeds=[901],duration_s=1.)
                with isolated_random_state():
                    evaluate(monitoring,"compressed_async_age",7,checkpoint,path/f"monitor_{saved['stage']}")
            resumed=train(config,"compressed_async_age",7,path/"resumed",boundary,on_stage=monitor)
            a=torch.load(uninterrupted,weights_only=False); b=torch.load(resumed,weights_only=False)
            self.assertTrue(all(not transition.done for transition in a["replay"]))
            self.assertEqual(sum(transition.truncated for transition in a["replay"]),2)
            self.assertEqual(a["checkpoint_schema"],3)
            self.assert_nested_equal(a,b)
            for name in ("steps.csv","episodes.csv","losses.csv","replay_audit.csv"):
                self.assertEqual((uninterrupted.parent/name).read_bytes(),(resumed.parent/name).read_bytes())


if __name__=="__main__": unittest.main()
