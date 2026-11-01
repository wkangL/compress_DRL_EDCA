from __future__ import annotations
import torch
from torch import nn


class CausalBlock(nn.Module):
    def __init__(self, input_dim: int, channels: int, kernel: int, dilation: int):
        super().__init__()
        self.pad=(kernel-1)*dilation
        self.conv=nn.Conv1d(input_dim,channels,kernel,padding=self.pad,dilation=dilation)
        self.skip=nn.Conv1d(input_dim,channels,1) if input_dim!=channels else nn.Identity()
        self.activation=nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y=self.conv(x)
        if self.pad: y=y[...,:-self.pad]
        return self.activation(y+self.skip(x))


class StaEncoder(nn.Module):
    """One shared causal TCN for all STAs and ACs, input (..., time, 6)."""
    def __init__(self, channels: int, kernel_size: int, dilations: list[int], latent_dim: int):
        super().__init__()
        self.blocks=nn.Sequential(*[CausalBlock(6 if i==0 else channels,channels,kernel_size,d)
                                    for i,d in enumerate(dilations)])
        self.output=nn.Linear(channels,latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape=x.shape[:-2]
        y=x.reshape(-1,x.shape[-2],6).transpose(1,2)
        y=self.blocks(y).mean(dim=-1)
        return self.output(y).reshape(*shape,-1)
