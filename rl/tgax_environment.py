"""BSS-0 online observation and reward adapter for the persistent TGax backend."""
from __future__ import annotations
from pathlib import Path
import numpy as np

from reporting.sta_state import StaHistory
from reporting.report_cache import ReportCache
from reporting.report_generator import ReportGenerator
from sim.backend.base_backend import ControlAction
from sim.backend.tgax_ns3_backend import TgaxNs3Backend
from sim.statistics.ap_state import age_statistics
from rl.tgax_reward import tgax_reward
from rl.cmdp_reward import (LagrangeController, cmdp_components, directional_cmdp_components,
                            completed_qos_increment, five_cost_cmdp_components, lagrangian_reward, legacy_cmdp_components_v1,
                            three_cost_cmdp_components, two_cost_cmdp_components)


CMDP_VERSIONS = {"tgax_cmdp_v1", "tgax_cmdp_v2", "tgax_cmdp_directional_v3",
                 "tgax_cmdp_two_cost_v4", "tgax_cmdp_three_cost_v5", "tgax_cmdp_five_cost_v6"}
EXACT_QOS_VERSIONS = {"tgax_cmdp_two_cost_v4", "tgax_cmdp_three_cost_v5", "tgax_cmdp_five_cost_v6"}


class TgaxControlEnvironment:
    def __init__(self,config:dict,method:str,seed:int,directory:Path,duration_s:float,encoder_version:int=0):
        self.config=config; self.method=method; self.time=0.; self.duration_s=duration_s
        n=config["network"]["num_sta"]; window=round(config["timing"]["history_s"]/config["timing"]["sta_sample_s"])
        self.history=StaHistory(n,window); self.cache=ReportCache(n,window,encoder_version)
        self.generator=ReportGenerator(self.history,self.cache)
        self.backend=TgaxNs3Backend(config,seed,directory,duration_s)
        self.ap_queue=np.zeros(4)
        self.action=np.asarray(config["action"].get("default_action",[0,0,5,5,2]),dtype=np.int64)
        self.previous_report_count=0; self.demand_debt=np.zeros(4)
        self.sta_demand_debt=np.zeros(n)
        self.low_zero_streak=np.zeros(2,dtype=np.int64)
        self.lagrange=None
        if config["reward"].get("version") in CMDP_VERSIONS and config["reward"].get("budgets") is not None:
            reward=config["reward"]
            self.lagrange=LagrangeController(np.asarray(reward["budgets"],dtype=float),
                np.asarray(reward["multiplier_initial"],dtype=float),reward["multiplier_learning_rate"],
                reward["multiplier_ema_alpha"],reward["multiplier_maximum"])
        self.generator.process(self.backend.warmup_events)
        for event in self.backend.warmup_events:
            if event.kind=="U":
                self.demand_debt[int(event.values[0])]+=event.values[1]-event.values[2]
            elif event.kind=="V":
                self.sta_demand_debt[int(event.values[0])]+=event.values[1]-event.values[2]
            elif event.kind=="S" and int(event.values[1])==0:
                self.ap_queue[int(event.values[2])]=event.values[3]
        self.demand_debt=np.maximum(self.demand_debt,0)
        self.sta_demand_debt=np.maximum(self.sta_demand_debt,0)

    def action_features(self,action):
        c=self.config["action"]
        profile_count=int(c.get("profile_branch_size",16))
        one=np.eye(profile_count,dtype=np.float32)[action[0]]; values=[]
        for index,key in enumerate(("alpha","beta1","beta2","gamma"),1):
            options=c[key]; values.append((options[action[index]]-min(options))/(max(options)-min(options)))
        return np.r_[one,values].astype(np.float32)

    def decode(self,a,custom=True):
        c=self.config["action"]
        fallback=c.get("default_fallback_index")
        native=bool(fallback is not None and int(a[0])==int(fallback))
        profile=int(c.get("default_profile",0)) if native else int(a[0])
        return ControlAction(profile,custom and not native,c["alpha"][a[1]],c["beta1"][a[2]],c["beta2"][a[3]],c["gamma"][a[4]])

    def observation(self,events,start):
        history,age,valid=self.cache.snapshot(self.time)
        t=self.config["state"]; busy=[]; ul=np.zeros(4)
        for e in events:
            if e.kind=="B":
                left,right=max(start,e.values[0]),min(self.time,e.values[0]+e.values[1])
                if right>left: busy.append((left,right))
            elif e.kind=="M" and int(e.values[0])==0:
                ul[int(e.values[1])]+=e.values[8]
            elif e.kind=="S" and int(e.values[1])==0:
                self.ap_queue[int(e.values[2])]=e.values[3]
        busy.sort(); covered=0.; last=start
        for left,right in busy: covered+=max(0,right-max(left,last)); last=max(last,right)
        duration=max(self.time-start,1e-9); scale=t["goodput_scale_mbps"]
        global_stats=np.asarray([covered/duration,*list(ul*8/1e6/duration/scale),
            *list(np.log1p(self.ap_queue)/np.log1p(t["queue_byte_scale"])),
            min(self.history.num_sta/t["associated_sta_scale"],1.)],dtype=np.float32)
        return dict(history=history,age=np.minimum(age,t["age_clip_s"])/t["age_scale_s"],valid=valid,
            global_stats=global_stats,action=self.action_features(self.action),report_time=self.time-age,
            decision_time=np.asarray(self.time),encoder_version=self.cache.encoder_version)

    def reset_observation(self): return self.observation([],0.)

    def step(self,action,duration=1.,custom=True):
        previous=self.action.copy(); start=self.time
        events=self.backend.step(self.decode(action,custom),duration); self.time=self.backend.time
        self.generator.process(events); self.action=np.asarray(action,dtype=np.int64).copy()
        delays=[[[] for _ in range(4)] for _ in range(2)]; ledgers={}; generated=np.zeros(4); received=np.zeros(4)
        submitted=np.zeros(4)
        sta_generated=np.zeros(self.history.num_sta); sta_received=np.zeros(self.history.num_sta)
        for e in events:
            v=e.values
            if e.kind=="Y": delays[int(v[1])][int(v[2])].append(v[3])
            elif e.kind=="M": ledgers[int(v[0]),int(v[1])]=v[2:]
            elif e.kind=="U":
                generated[int(v[0])]=v[1]; received[int(v[0])]=v[2]
                if len(v)>3: submitted[int(v[0])]=v[3]
            elif e.kind=="V": sta_generated[int(v[0])]=v[1]; sta_received[int(v[0])]=v[2]
        if len(ledgers)!=8: raise RuntimeError("Incomplete TGax direction/AC ledger")
        available=self.demand_debt+generated
        satisfaction=np.divide(received,available,out=np.zeros(4),where=available>0).clip(0,1)
        active=available>0; self.demand_debt=np.maximum(available-received,0)
        sta_available=self.sta_demand_debt+sta_generated
        sta_satisfaction=np.divide(sta_received,sta_available,out=np.zeros_like(sta_available),
                                   where=sta_available>0).clip(0,1)
        sta_active=sta_available>0
        active_values=sta_satisfaction[sta_active]
        square_sum=float(np.square(active_values).sum())
        jain=(float(active_values.sum()**2/(len(active_values)*square_sum))
              if len(active_values) and square_sum>0 else 0.)
        self.sta_demand_debt=np.maximum(sta_available-sta_received,0)
        metrics={"application_generated_bytes":generated.tolist(),"application_submitted_bytes":submitted.tolist(),
            "application_received_bytes":received.tolist(),
            "application_demand_debt_bytes":self.demand_debt.tolist(),"application_satisfaction":satisfaction.tolist(),
            "application_demand_active":active.tolist(),
            "sta_application_generated_bytes":sta_generated.tolist(),
            "sta_application_received_bytes":sta_received.tolist(),
            "sta_application_demand_debt_bytes":self.sta_demand_debt.tolist(),
            "sta_application_satisfaction":sta_satisfaction.tolist(),
            "sta_application_demand_active":sta_active.tolist(),
            "jain_demand_satisfaction":jain,
            "jain_active_sta_count":int(sta_active.sum())}
        metrics["demand_satisfaction"]=metrics["application_satisfaction"]
        metrics["demand_active"]=metrics["application_demand_active"]
        for offset,ac in enumerate((2,3)):
            self.low_zero_streak[offset]=(self.low_zero_streak[offset]+1
                if active[ac] and received[ac]<=0 else 0)
        metrics["be_or_bk_zero_service_streak_violation"]=bool((self.low_zero_streak>=3).any())
        total_attempts=total_retries=total_failed=total_risk=0
        for ac,name in enumerate(("vo","vi","be","bk")):
            eligible=ontime=0
            for direction,dname in enumerate(("ul","dl")):
                x=ledgers[direction,ac]
                arrived,acked,failed,pstart,pend,arrived_b,acked_b,failed_b,attempts,retries,de,ot=map(int,x)
                if pstart+arrived != acked+failed+pend:
                    raise RuntimeError(f"TGax MAC ledger mismatch {dname}/{name}: {x}")
                if ot>de: raise RuntimeError("TGax on-time ledger mismatch")
                samples=delays[direction][ac]
                metrics.update({f"{dname}_{name}_arrived":arrived,f"{dname}_{name}_acked":acked,
                    f"{dname}_{name}_failed":failed,f"{dname}_{name}_pending_end":pend,
                    f"{dname}_{name}_pending_start":pstart,
                    f"{dname}_{name}_tx_attempts":attempts,f"{dname}_{name}_retransmissions":retries,
                    f"{dname}_{name}_goodput_mbps":acked_b*8/1e6/duration,
                    f"{dname}_{name}_delay_p95_ms":float(np.percentile(samples,95)) if samples else float("nan"),
                    f"{dname}_{name}_delay_samples_ms":samples,
                    f"{dname}_{name}_delay_valid":bool(samples),
                    f"{dname}_{name}_final_failure_rate":failed/max(pstart+arrived,1),
                    f"{dname}_{name}_retransmission_rate":retries/max(attempts,1)})
                if self.config["reward"].get("version") in EXACT_QOS_VERSIONS and ac < 2:
                    if len(samples) != acked:
                        raise RuntimeError(f"ACK/delay sample mismatch {dname}/{name}: {acked} vs {len(samples)}")
                    reference=float(self.config["reward"]["qos_delay_reference_ms"][ac])
                    increment,count=completed_qos_increment(samples,failed,reference)
                    prefix=f"qos_{name}_{dname}"
                    metrics.update({prefix:(float(increment/count) if count else None),
                                    f"{prefix}_valid":bool(count),
                                    f"{prefix}_finalized":count,
                                    f"{prefix}_pending_end":pend})
                total_attempts+=attempts; total_retries+=retries; total_failed+=failed; total_risk+=pstart+arrived
                eligible+=de; ontime+=ot
            metrics[f"{name}_on_time_delivery_rate"]=ontime/max(eligible,1)
            metrics[f"{name}_on_time_delivery_valid"]=bool(eligible)
            metrics[f"{name}_deadline_eligible"]=eligible
            metrics[f"{name}_on_time_delivered"]=ontime
        metrics["retransmission_rate"]=total_retries/max(total_attempts,1)
        metrics["final_failure_fraction"]=total_failed/max(total_risk,1)
        _,age,valid=self.cache.snapshot(self.time); mean,p95,coverage=age_statistics(age,valid)
        metrics.update(report_age_mean_s=mean,report_age_p95_s=p95,valid_report_coverage=coverage)
        scalar_start=int(self.config["action"].get("profile_branch_size",16))
        change=(int(previous[0]!=self.action[0])+np.abs(self.action_features(previous)[scalar_start:]-self.action_features(self.action)[scalar_start:]).sum())/5
        reward_version = self.config["reward"].get("version")
        if reward_version in CMDP_VERSIONS:
            component_fn = ({"tgax_cmdp_v2": cmdp_components,
                             "tgax_cmdp_directional_v3": directional_cmdp_components,
                             "tgax_cmdp_two_cost_v4": two_cost_cmdp_components,
                             "tgax_cmdp_three_cost_v5": three_cost_cmdp_components,
                             "tgax_cmdp_five_cost_v6": five_cost_cmdp_components}
                            .get(reward_version, legacy_cmdp_components_v1))
            utility,costs,parts=component_fn(metrics,self.config["reward"])
            value=(utility if self.lagrange is None else
                   lagrangian_reward(utility,costs,self.lagrange.budgets,self.lagrange.multipliers,
                                     excess_only=(reward_version == "tgax_cmdp_v1")))
            metrics.update(cmdp_costs=costs.tolist(),
                cmdp_multipliers=(self.lagrange.multipliers.tolist() if self.lagrange is not None else None),
                reward=value)
            metrics.update({f"cmdp_{k}":v for k,v in parts.items()})
            if self.lagrange is not None:
                metrics.update({f"lambda_{name}":float(self.lagrange.multipliers[index])
                                for index,name in enumerate(parts["cost_order"])})
        else:
            value,parts=tgax_reward(metrics,float(change),self.config["reward"])
            metrics.update(reward=value,**{f"reward_{k}":v for k,v in parts.items()})
        return self.observation(events,start),value,metrics

    def update_lagrange(self,episode_costs,valid_mask=None):
        if self.lagrange is None: raise RuntimeError("CMDP budgets are not configured")
        return self.lagrange.update(episode_costs,valid_mask)

    def close(self): self.backend.close()
