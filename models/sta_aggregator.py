from __future__ import annotations
import torch
from torch import nn


class StaAggregator(nn.Module):
    """Permutation-invariant per-AC attention + max; empty sets return finite zeros."""
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.shared=nn.Sequential(nn.Linear(input_dim+2,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        self.attention=nn.Linear(hidden,1)
        self.output_dim=4*(2*hidden+3)

    def forward(self, z: torch.Tensor, ages: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # z: batch, STA, AC, latent. Statistics exclude never-received reports.
        b,n,ac,_=z.shape
        v=valid[:,:,None,None].expand(b,n,ac,1)
        a=ages[:,:,None,None].expand(b,n,ac,1)
        h=self.shared(torch.cat([z,a,v],dim=-1))
        mask=v.bool()
        scores=self.attention(h).masked_fill(~mask,-1e9)
        weights=torch.softmax(scores,dim=1)*v
        weights=weights/weights.sum(dim=1,keepdim=True).clamp_min(1e-9)
        pooled=(weights*h).sum(dim=1)
        maximum=h.masked_fill(~mask,-1e9).amax(dim=1)
        nonempty=valid.sum(dim=1)>0
        maximum=torch.where(nonempty[:,None,None],maximum,torch.zeros_like(maximum))
        count=valid.sum(dim=1).clamp_min(1)
        mean=(ages*valid).sum(dim=1)/count
        # Masked P95 with deterministic linear interpolation, invariant to STA order.
        ordered=ages.masked_fill(~valid.bool(),float("inf")).sort(dim=1).values
        rank=.95*(count-1)
        low=ordered.gather(1,rank.floor().long()[:,None]).squeeze(1)
        high=ordered.gather(1,rank.ceil().long()[:,None]).squeeze(1)
        p95=torch.where(nonempty,low+(high-low)*(rank-rank.floor()),torch.zeros_like(low))
        coverage=valid.mean(dim=1)
        stats=torch.stack([mean,p95,coverage],dim=-1)[:,None,:].expand(b,ac,3)
        return torch.cat([pooled,maximum,stats],dim=-1).flatten(1)
