from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Report:
    sta: int
    generated_s: float
    received_s: float
    txop_id: int
    encoder_version: int
    history: np.ndarray


class ReportCache:
    def __init__(self, num_sta: int, window: int, encoder_version: int):
        self.num_sta=num_sta; self.window=window; self.encoder_version=encoder_version
        self.reports: dict[int,Report]={}

    def receive(self, report: Report) -> bool:
        if report.encoder_version!=self.encoder_version:
            raise ValueError("Mixed encoder versions are prohibited")
        if report.received_s<report.generated_s or not 0<=report.sta<self.num_sta:
            raise ValueError("Invalid report identity or timestamp")
        if report.history.shape!=(4,self.window,6): raise ValueError("Wrong history dimensions")
        old=self.reports.get(report.sta)
        if old and report.generated_s<=old.generated_s: return False
        self.reports[report.sta]=Report(report.sta,report.generated_s,report.received_s,
                                      report.txop_id,report.encoder_version,report.history.copy())
        return True

    def sync_version(self, version: int) -> None:
        if version<=self.encoder_version: raise ValueError("Version must increase")
        self.encoder_version=version; self.reports.clear()

    def snapshot(self, now: float) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        histories=np.zeros((self.num_sta,4,self.window,6),dtype=np.float32)
        age=np.zeros(self.num_sta,dtype=np.float32); valid=np.zeros(self.num_sta,dtype=np.float32)
        for sta,r in self.reports.items():
            if r.received_s>now: raise ValueError("Decision cannot see a future report")
            histories[sta]=r.history; age[sta]=now-r.generated_s; valid[sta]=1
        return histories,age,valid
