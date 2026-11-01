from __future__ import annotations
from pathlib import Path
import numpy as np
from project_config import ROOT,load_config
from sim.backend.ns3_backend import Ns3Backend
from sim.backend.base_backend import ControlAction
from sim.statistics.metrics import summarize
from sim.statistics.ap_state import ap_features,age_statistics
from reporting.sta_state import StaHistory
from reporting.report_cache import ReportCache
from reporting.report_generator import ReportGenerator
from rl.reward import reward


class ControlEnvironment:
    """The only RL-facing adapter; method selection changes information, not MAC rules."""
    def __init__(self, config: dict, method: str, scenario: dict, seed: int,
                 directory: Path, capacity: float, encoder_version: int):
        self.config=config; self.method=method; self.capacity=capacity
        n=config["network"]["num_sta"]; window=round(config["timing"]["history_s"]/config["timing"]["sta_sample_s"])
        self.history=StaHistory(n,window); self.cache=ReportCache(n,window,encoder_version)
        self.generator=ReportGenerator(self.history,self.cache)
        self.backend=Ns3Backend(config,scenario,seed,directory)
        self.ap_queue=np.zeros(4); self.time=0.
        self.action=np.array([0,0,5,5,2],dtype=np.int64)
        self.previous_report_count=0

    def action_features(self, action: np.ndarray) -> np.ndarray:
        c=self.config["action"]; one=np.eye(16,dtype=np.float32)[action[0]]
        values=[]
        for index,key in enumerate(("alpha","beta1","beta2","gamma"),1):
            options=c[key]; values.append((options[action[index]]-min(options))/(max(options)-min(options)))
        return np.r_[one,values].astype(np.float32)

    def decode(self, a: np.ndarray) -> ControlAction:
        c=self.config["action"]
        return ControlAction(int(a[0]),self.method!="default",c["alpha"][a[1]],c["beta1"][a[2]],c["beta2"][a[3]],c["gamma"][a[4]])

    def observation(self, period_events: list, start: float) -> dict:
        fresh=self.method in ("full_fresh","compressed_fresh")
        if fresh:
            histories=self.history.all(); ages=np.zeros(self.history.num_sta,dtype=np.float32)
            valid=np.zeros(self.history.num_sta,dtype=np.float32)
            valid[:round(self.config["network"]["enhanced_fraction"]*len(valid))]=1
        else: histories,ages,valid=self.cache.snapshot(self.time)
        use_age=self.method!="compressed_async_no_age"
        t=self.config["state"]; scale=t["age_scale_s"]
        period=ap_features(period_events,start,self.time,self.capacity,self.ap_queue,
                           self.history.num_sta,t["associated_sta_scale"],t["queue_byte_scale"])
        return dict(history=histories,age=np.minimum(ages,t["age_clip_s"])/scale if use_age else np.zeros_like(ages),
                    valid=valid,global_stats=period,action=self.action_features(self.action),
                    report_time=self.time-ages,decision_time=np.asarray(self.time),encoder_version=self.cache.encoder_version)

    def reset_observation(self) -> dict:
        return self.observation([],0.)

    def step(self, action: np.ndarray, duration: float) -> tuple[dict,float,dict]:
        previous=self.action.copy(); start=self.time
        events=self.backend.step(self.decode(action),duration); self.time=self.backend.time
        self.last_events=events
        self.generator.process(events)
        for e in events:
            if e.kind=="S" and e.values[1]==0: self.ap_queue[int(e.values[2])]=e.values[3]
        self.action=action.copy()
        metrics=summarize(events,self.history.num_sta,duration)
        _,age,valid=self.cache.snapshot(self.time)
        mean,p95,coverage=age_statistics(age,valid)
        if self.method in ("full_fresh","compressed_fresh"):
            mean=p95=0.; coverage=self.config["network"]["enhanced_fraction"]
        if self.method=="default": mean=p95=coverage=0.
        metrics.update(report_age_mean_s=mean,report_age_p95_s=p95,
                       report_age_max_s=float(age[valid>0].max()) if valid.any() else 0.,valid_report_coverage=coverage)
        components=(4*self.config["model"]["latent_dim"] if self.method.startswith("compressed") else 4*self.history.window*6)
        count=self.generator.received-self.previous_report_count; self.previous_report_count=self.generator.received
        if self.method in ("full_fresh","compressed_fresh"): count=round(coverage*self.history.num_sta)
        if self.method=="default": count=0
        metrics.update(report_rate_per_sta=count/self.history.num_sta/duration,
                       feedback_components_per_s=count*components/duration,fp32_equivalent_bytes_per_s=count*components*4/duration)
        change=(int(previous[0]!=action[0])+np.abs(self.action_features(previous)[16:]-self.action_features(action)[16:]).sum())/5
        value,parts=reward(metrics,float(change),self.config["reward"])
        metrics.update(reward=value,**{f"reward_{k}":v for k,v in parts.items()})
        return self.observation(events,start),value,metrics

    def close(self) -> None: self.backend.close()
