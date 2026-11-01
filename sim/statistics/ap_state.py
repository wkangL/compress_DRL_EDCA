from __future__ import annotations
import numpy as np
from sim.backend.base_backend import Event

GLOBAL_DIM=10
GLOBAL_FEATURES=("busy_fraction", "ul_vo_goodput", "ul_vi_goodput", "ul_be_goodput",
                 "ul_bk_goodput", "ap_vo_queue", "ap_vi_queue", "ap_be_queue",
                 "ap_bk_queue", "associated_sta_count")


def age_statistics(age: np.ndarray, valid: np.ndarray) -> tuple[float,float,float]:
    a=age[valid>0]
    return (float(a.mean()) if len(a) else 0.,float(np.percentile(a,95)) if len(a) else 0.,float(valid.mean()))


def ap_features(events: list[Event], start: float, end: float, capacity_mbps: float,
                queue: np.ndarray, associated_sta: int, associated_scale: float,
                queue_byte_scale: float) -> np.ndarray:
    """Ten AP-local features with one decision window and no feedback-age duplication."""
    if end<start or capacity_mbps<=0 or associated_scale<=0 or queue_byte_scale<=0:
        raise ValueError("AP feature scales and window must be positive")
    ul_ac_bytes=np.zeros(4); busy=[]
    for e in events:
        v=e.values
        if e.kind=="B":
            left,right=max(start,v[0]),min(end,v[0]+v[1])
            if right>left: busy.append((left,right))
        elif start<v[0]<=end:
            if e.kind=="D":
                _,i,ac,size,delay,node=v
                if node>0: ul_ac_bytes[int(ac)]+=size
    busy.sort(); covered=0.; last=start
    for left,right in busy:
        covered+=max(0,right-max(left,last)); last=max(last,right)
    duration=max(end-start,1e-9); rates=ul_ac_bytes*8/1e6/duration
    return np.asarray([covered/duration,*list(rates/capacity_mbps),
                       *list(np.log1p(queue)/np.log1p(queue_byte_scale)),
                       min(associated_sta/associated_scale,1.)],dtype=np.float32)
