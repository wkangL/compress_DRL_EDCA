import unittest
import numpy as np
import torch
from copy import deepcopy
from project_config import ROOT,load_config
from rl.reward import reward
from rl.replay_buffer import ReplayBuffer,Transition
from rl.trainer import td_update,freeze_target,update_target
from models.branching_ddqn import ControlNetwork
from reporting.sta_state import StateScaler
from sim.statistics.ap_state import GLOBAL_DIM


class RlTests(unittest.TestCase):
    def test_reward_inactive_ac_and_time_normalization(self) -> None:
        c=load_config(ROOT/"configs/quick.yaml")["reward"]
        m={"collision_rate":0.,"drop_rate":0.,"jain_fairness":1.}
        for ac in ("vo","vi","be","bk"):
            m.update({f"{ac}_offered_mbps":0.,f"{ac}_goodput_mbps":0.,f"{ac}_delay_valid":False,f"{ac}_delay_p95_ms":0.})
        m["be_offered_mbps"]=5; m["be_goodput_mbps"]=5
        self.assertAlmostEqual(reward(m,0,c)[0],1.5)
        m["be_goodput_mbps"]=2.5
        self.assertAlmostEqual(reward(m,0,c)[0],.75)

    def test_reward_v2_uses_causal_deadlines_and_service_ablation(self) -> None:
        base=load_config(ROOT/"configs/reward_v2.yaml")["reward"]
        m={"collision_rate":.1,"drop_rate":.9,"final_failure_fraction":.2,
           "jain_fairness":.8,"vo_on_time_delivery_valid":True,
           "vo_on_time_delivery_rate":.75,"vi_on_time_delivery_valid":True,
           "vi_on_time_delivery_rate":.5}
        for ac in ("vo","vi","be","bk"):
            m.update({f"{ac}_offered_mbps":1.,f"{ac}_goodput_mbps":1.})
        value,parts=reward(m,.5,base)
        self.assertAlmostEqual(value,1+.8*.75+1*.5-.3*.1-.3*.2-.2*.2-.02*.5)
        self.assertEqual(parts["low_priority_deficit"],0.)
        service=load_config(ROOT/"configs/reward_v2_service.yaml")["reward"]
        m["be_goodput_mbps"]=.4; m["bk_goodput_mbps"]=.8
        protected,parts=reward(m,.5,service)
        # floor=.9 gives mean deficit (.5+.1)/2=.3 and weight=.5.
        unprotected=reward(m,.5,base)[0]
        self.assertAlmostEqual(parts["low_priority_deficit"],.3)
        self.assertAlmostEqual(protected,unprotected-.15)

    def test_reward_v2_never_treats_immature_packets_as_misses(self) -> None:
        c=load_config(ROOT/"configs/reward_v2.yaml")["reward"]
        m={"collision_rate":0.,"drop_rate":0.,"final_failure_fraction":0.,"jain_fairness":1.,
           "vo_on_time_delivery_valid":False,"vo_on_time_delivery_rate":0.,
           "vi_on_time_delivery_valid":False,"vi_on_time_delivery_rate":0.}
        for ac in ("vo","vi","be","bk"):
            m.update({f"{ac}_offered_mbps":0.,f"{ac}_goodput_mbps":0.})
        value,parts=reward(m,0,c)
        self.assertEqual(value,0.)
        self.assertEqual(parts["on_time_vo"],0.)

    def test_replay_is_a_snapshot_and_td_is_finite(self) -> None:
        torch.set_num_threads(2); c=load_config(ROOT/"configs/quick.yaml")
        s=dict(history=np.ones((2,4,10,6),dtype=np.float32),age=np.zeros(2),valid=np.ones(2),global_stats=np.zeros(GLOBAL_DIM),action=np.zeros(20))
        replay=ReplayBuffer(10,1)
        t=Transition(s,np.array([0,0,5,5,2]),1.,deepcopy(s),True,0)
        replay.add(t); s["history"][:]=99
        self.assertEqual(replay.data[0].state["history"].max(),1.)
        scaler=StateScaler(100); scaler.fit(np.stack([np.zeros((10,6)),np.ones((10,6))]))
        model=ControlNetwork(c,"compressed_async_age",GLOBAL_DIM); target=freeze_target(deepcopy(model))
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        stats=td_update(model,target,replay.sample(1),scaler,opt,c["training"])
        self.assertTrue(np.isfinite(stats["td_loss"]))

    def test_time_limit_bootstraps_and_target_is_independent_frozen(self) -> None:
        c=load_config(ROOT/"configs/quick.yaml"); scaler=StateScaler(100)
        scaler.fit(np.stack([np.zeros((10,6)),np.ones((10,6))]))
        state=dict(history=np.ones((2,4,10,6),dtype=np.float32),age=np.zeros(2),valid=np.ones(2),
                   global_stats=np.zeros(GLOBAL_DIM),action=np.zeros(20))
        transition=Transition(state,np.array([0,0,5,5,2]),1.,deepcopy(state),False,0,truncated=True)
        model=ControlNetwork(c,"compressed_async_age",GLOBAL_DIM); target=freeze_target(deepcopy(model))
        self.assertTrue(all(not p.requires_grad for p in target.parameters()))
        self.assertTrue(all(a.data_ptr()!=b.data_ptr() for a,b in zip(model.parameters(),target.parameters())))
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        stats=td_update(model,target,[transition],scaler,opt,c["training"])
        self.assertEqual(stats["bootstrap_fraction"],1.)
        self.assertEqual(stats["terminal_fraction"],0.)
        self.assertEqual(stats["truncated_fraction"],1.)
        self.assertTrue(all(p.grad is None for p in target.parameters()))

    def test_hard_and_soft_target_updates(self) -> None:
        c=load_config(ROOT/"configs/quick.yaml")["training"]
        model=ControlNetwork(load_config(ROOT/"configs/quick.yaml"),"compressed_async_age",GLOBAL_DIM)
        target=freeze_target(deepcopy(model))
        with torch.no_grad(): next(model.parameters()).add_(1.)
        hard=dict(c,target_update_mode="hard",target_update_steps=2)
        changed,_,after=update_target(model,target,1,hard); self.assertFalse(changed); self.assertGreater(after,0)
        changed,_,after=update_target(model,target,2,hard); self.assertTrue(changed); self.assertEqual(after,0)
        with torch.no_grad(): next(model.parameters()).add_(1.)
        soft=dict(c,target_update_mode="soft",target_tau=.5)
        changed,before,after=update_target(model,target,3,soft)
        self.assertTrue(changed); self.assertGreater(before,after); self.assertGreater(after,0)


if __name__=="__main__": unittest.main()
