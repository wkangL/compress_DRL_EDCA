from __future__ import annotations
from collections import deque
import numpy as np


class StaHistory:
    """Per-AC interval features. No cumulative counters are accepted here."""
    def __init__(self, num_sta: int, window: int):
        self.num_sta=num_sta; self.window=window
        self.data=[[deque(maxlen=window) for _ in range(4)] for _ in range(num_sta)]
        self.last=np.full((num_sta,4),-np.inf)

    def append(self, sta: int, ac: int, time_s: float, features: np.ndarray) -> None:
        x=np.asarray(features,dtype=np.float32).copy()
        if x.shape!=(6,) or not np.isfinite(x).all() or (x<0).any() or x[5] not in (0,1):
            raise ValueError("Expected six finite nonnegative interval features")
        if time_s<=self.last[sta,ac]: raise ValueError("Samples must be chronological")
        if x[5]==0: x[1]=0
        self.last[sta,ac]=time_s; self.data[sta][ac].append(x)

    def snapshot(self, sta: int) -> np.ndarray:
        out=np.zeros((4,self.window,6),dtype=np.float32)
        for ac in range(4):
            values=self.data[sta][ac]
            if values: out[ac,-len(values):]=np.asarray(values)
        return out

    def all(self) -> np.ndarray:
        return np.stack([self.snapshot(i) for i in range(self.num_sta)])


class StateScaler:
    """Fit once from training samples; validity stays binary and invalid delay stays zero."""
    def __init__(self, delay_clip_ms: float):
        self.delay_clip_ms=delay_clip_ms
        self.mean=np.zeros(5,dtype=np.float32); self.std=np.ones(5,dtype=np.float32)
        self.fitted=False

    def _log(self, x: np.ndarray) -> np.ndarray:
        y=np.asarray(x,dtype=np.float32).copy()
        y[...,1]=np.minimum(y[...,1],self.delay_clip_ms)
        y[...,:5]=np.log1p(y[...,:5])
        return y

    def fit(self, samples: np.ndarray, split: str="train") -> None:
        if split!="train" or self.fitted: raise ValueError("Scaler can only be fitted once on training data")
        x=self._log(samples).reshape(-1,6)
        for f in range(5):
            v=x[x[:,5]>0,f] if f==1 else x[:,f]
            if len(v): self.mean[f]=v.mean(); self.std[f]=max(float(v.std()),1e-6)
        self.fitted=True

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted: raise ValueError("Training scaler is not fitted")
        y=self._log(x); y[...,:5]=(y[...,:5]-self.mean)/self.std
        y[...,1]=np.where(y[...,5]>0,y[...,1],0)
        return y
