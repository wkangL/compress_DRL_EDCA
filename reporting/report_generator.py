from __future__ import annotations
import numpy as np
from reporting.sta_state import StaHistory
from reporting.report_cache import Report,ReportCache
from sim.backend.base_backend import Event


class ReportGenerator:
    """G snapshots the window before TX; R releases it only after an actual ACK."""
    def __init__(self, histories: StaHistory, cache: ReportCache):
        self.histories=histories; self.cache=cache
        self.pending: dict[tuple[int,int],tuple[float,np.ndarray]]={}
        self.received=0

    def process(self, events: list[Event]) -> None:
        for e in events:
            v=e.values
            if e.kind=="S" and v[1]>0:
                self.histories.append(int(v[1])-1,int(v[2]),v[0],np.asarray(v[3:9]))
            elif e.kind=="G":
                t,sta,txop=v
                self.pending[int(sta),int(txop)]=(t,self.histories.snapshot(int(sta)))
            elif e.kind=="R":
                received,sta,generated,txop=v; key=(int(sta),int(txop))
                if key not in self.pending: raise ValueError("Report ACK without a generation snapshot")
                t,x=self.pending.pop(key)
                if abs(t-generated)>1e-8: raise ValueError("Report timestamp mismatch")
                accepted=self.cache.receive(Report(int(sta),generated,received,int(txop),self.cache.encoder_version,x))
                self.received+=int(accepted)
            elif e.kind=="X":
                self.pending.pop((int(v[1]),int(v[2])),None)
