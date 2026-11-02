from __future__ import annotations
import torch
from torch import nn
from models.sta_encoder import StaEncoder
from models.sta_aggregator import StaAggregator


class BranchingQ(nn.Module):
    """Shared V plus five independent centered advantages; no Cartesian action output."""
    def __init__(self, input_dim: int, widths: list[int], branches: list[int]):
        super().__init__()
        layers=[]; previous=input_dim
        for width in widths: layers.extend([nn.Linear(previous,width),nn.ReLU()]); previous=width
        self.backbone=nn.Sequential(*layers)
        self.value=nn.Linear(previous,1)
        self.advantages=nn.ModuleList([nn.Linear(previous,size) for size in branches])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        h=self.backbone(x); value=self.value(h)
        return [value+(a:=branch(h))-a.mean(dim=-1,keepdim=True) for branch in self.advantages]


class ControlNetwork(nn.Module):
    def __init__(self, config: dict, method: str, global_dim: int):
        super().__init__()
        m=config["model"]; action=config["action"]
        self.method=method
        self.compressed=method.startswith("compressed")
        steps=round(config["timing"]["history_s"]/config["timing"]["sta_sample_s"])
        self.encoder=StaEncoder(m["channels"],m["kernel_size"],m["dilations"],m["latent_dim"]) if self.compressed else nn.Flatten(start_dim=-2)
        latent=m["latent_dim"] if self.compressed else steps*6
        self.aggregator=StaAggregator(latent,m["aggregation_dim"])
        self.profile_branch_size=int(action.get("profile_branch_size",16))
        self.default_fallback_index=action.get("default_fallback_index")
        self.profile_application_mode=action.get("profile_application_mode","unified")
        if self.profile_application_mode not in ("unified","split_ap_sta"):
            raise ValueError("action.profile_application_mode must be unified or split_ap_sta")
        self.profile_branch_count=2 if self.profile_application_mode=="split_ap_sta" else 1
        self.scalar_offset=self.profile_branch_count
        self.branches=([self.profile_branch_size]*self.profile_branch_count+
                       [len(action[k]) for k in ("alpha","beta1","beta2","gamma")])
        self.action_feature_dim=self.profile_branch_size*self.profile_branch_count+4
        self.q=BranchingQ(global_dim+self.aggregator.output_dim+self.action_feature_dim,m["backbone"],self.branches)
        self.register_buffer("beta_mask",torch.tensor(action["beta2"])[None,:]>=torch.tensor(action["beta1"])[:,None])

    def forward(self, histories: torch.Tensor, ages: torch.Tensor, valid: torch.Tensor,
                global_stats: torch.Tensor,current_action: torch.Tensor,latent: torch.Tensor|None=None) -> list[torch.Tensor]:
        z=self.encoder(histories) if latent is None else latent
        pooled=self.aggregator(z,ages,valid)
        return self.q(torch.cat([global_stats,pooled,current_action],dim=-1))

    def greedy(self, q: list[torch.Tensor]) -> torch.Tensor:
        # Optimize the only coupled pair jointly over a 6x11 mask, not the full product.
        b1=self.scalar_offset+1; b2=self.scalar_offset+2
        pair=(q[b1][:,:,None]+q[b2][:,None,:]).masked_fill(~self.beta_mask,-torch.inf)
        index=pair.flatten(1).argmax(1)
        values=[q[i].argmax(1) for i in range(self.profile_branch_count)]
        values += [q[self.scalar_offset].argmax(1),index//q[b2].shape[1],
                   index%q[b2].shape[1],q[self.scalar_offset+3].argmax(1)]
        action=torch.stack(values,dim=1)
        if self.default_fallback_index is not None:
            fallback=(action[:,:self.profile_branch_count]==int(self.default_fallback_index)).all(dim=1)
            action[fallback,self.scalar_offset:]=0
        return action
