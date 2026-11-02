import unittest
from copy import deepcopy
import torch
from project_config import ROOT,load_config
from models.sta_encoder import StaEncoder
from models.sta_aggregator import StaAggregator
from models.branching_ddqn import ControlNetwork
from sim.statistics.ap_state import GLOBAL_DIM


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None: torch.set_num_threads(2)

    def test_permutation_and_empty_mask(self) -> None:
        torch.manual_seed(3); m=StaAggregator(8,32)
        z=torch.randn(2,10,4,8); a=torch.rand(2,10); v=torch.ones(2,10); v[0]=0; v[1,2]=0
        p=torch.randperm(10)
        first=m(z,a,v); second=m(z[:,p],a[:,p],v[:,p])
        self.assertTrue(torch.isfinite(first).all())
        torch.testing.assert_close(first,second,atol=1e-6,rtol=1e-5)
        self.assertEqual(float(first[0].detach().abs().sum()),0)

    def test_td_gradient_reaches_encoder(self) -> None:
        c=load_config(ROOT/"configs/quick.yaml"); m=ControlNetwork(c,"compressed_async_age",GLOBAL_DIM)
        x=torch.rand(2,3,4,10,6)
        q=m(x,torch.rand(2,3),torch.ones(2,3),torch.rand(2,GLOBAL_DIM),torch.rand(2,20))
        action=m.greedy(q)
        chosen=torch.stack([branch.gather(1,action[:,i,None]).squeeze(1) for i,branch in enumerate(q)],1)
        loss=torch.nn.functional.smooth_l1_loss(chosen,torch.ones_like(chosen))
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.encoder.parameters() if p.grad is not None),0)
        b1=[c["action"]["beta1"][i] for i in action[:,2]]
        b2=[c["action"]["beta2"][i] for i in action[:,3]]
        self.assertTrue(all(y>=x for x,y in zip(b1,b2)))

    def test_beta_mask_uses_joint_argmax(self) -> None:
        c=load_config(ROOT/"configs/quick.yaml"); m=ControlNetwork(c,"raw_async",GLOBAL_DIM)
        q=[torch.zeros(1,n) for n in m.branches]
        q[2][0,5]=100; q[3][0,0]=200
        a=m.greedy(q)[0]
        self.assertEqual(int(a[2]),0); self.assertEqual(int(a[3]),0)

    def test_explicit_default_fallback_is_canonical(self) -> None:
        c=deepcopy(load_config(ROOT/"configs/quick.yaml"))
        c["action"].update(profile_branch_size=17,default_fallback_index=16,
                           alpha=[1,2,3],beta1=[1.6,2.0],beta2=[2.0,2.4,2.8],gamma=[5,6,7])
        m=ControlNetwork(c,"compressed_async_age",GLOBAL_DIM)
        q=[torch.zeros(1,n) for n in m.branches]
        q[0][0,16]=10
        q[1][0,2]=5; q[2][0,1]=5; q[3][0,2]=5; q[4][0,2]=5
        action=m.greedy(q)
        self.assertEqual(action.tolist(),[[16,0,0,0,0]])
        self.assertEqual(m.action_feature_dim,21)

    def test_split_ap_sta_profiles_add_independent_branch(self) -> None:
        c=deepcopy(load_config(ROOT/"configs/quick.yaml"))
        c["action"].update(profile_application_mode="split_ap_sta",profile_branch_size=17,
                           default_fallback_index=16,default_action=[16,16,0,0,0,0],
                           alpha=[1,2,3],beta1=[1.6,2.0],beta2=[2.0,2.4,2.8],gamma=[5,6,7])
        m=ControlNetwork(c,"compressed_async_age",GLOBAL_DIM)
        self.assertEqual(m.branches,[17,17,3,2,3,3])
        self.assertEqual(m.action_feature_dim,38)
        q=[torch.zeros(1,n) for n in m.branches]
        q[0][0,16]=10; q[1][0,7]=10
        q[2][0,2]=5; q[3][0,1]=5; q[4][0,2]=5; q[5][0,2]=5
        self.assertEqual(m.greedy(q).tolist(),[[16,7,2,1,2,2]])
        q[1].zero_(); q[1][0,16]=10
        self.assertEqual(m.greedy(q).tolist(),[[16,16,0,0,0,0]])


if __name__=="__main__": unittest.main()
