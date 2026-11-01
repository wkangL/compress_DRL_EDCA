from __future__ import annotations
import numpy as np
from sim.backend.base_backend import Event


def summarize(events: list[Event], num_sta: int, duration: float) -> dict:
    """Summarize actual event records; timeout is explicitly not renamed collision."""
    ac_bytes=np.zeros(4); offered=np.zeros(4); sta_bytes=np.zeros(num_sta)
    direction_bytes=np.zeros((2,4)); direction_delays=[[[] for _ in range(4)] for _ in range(2)]
    direction_collisions=np.zeros((2,4),dtype=int); direction_other_phy=np.zeros((2,4),dtype=int)
    delays=[[] for _ in range(4)]
    attempts=retry=failed=discarded=timeouts=offered_packets=0
    reports=0; air_attempts=collisions=other_rx_failures=0
    records=[e.values for e in events if e.kind=="P"]
    for e in events:
        v=e.values
        if e.kind=="D":
            _,sta,ac,size,delay,node=v
            ac_bytes[int(ac)]+=size; sta_bytes[int(sta)]+=size; delays[int(ac)].append(delay)
            direction=0 if int(node)>0 else 1  # UL, DL
            direction_bytes[direction,int(ac)]+=size; direction_delays[direction][int(ac)].append(delay)
        elif e.kind=="S" and not records:
            offered[int(v[2])]+=v[9]; attempts+=v[10]; timeouts+=v[11]
            retry+=v[5]; failed+=v[6]; discarded+=v[7]
            if len(v)>13: offered_packets+=v[13]
        elif e.kind=="R": reports+=1
        elif e.kind=="A": air_attempts+=1
        elif e.kind=="C":
            collisions+=1
            if len(v)>=6 and int(v[4])>=0:
                direction_collisions[0 if int(v[1])==0 else 1,int(v[4])]+=1
        elif e.kind=="F" and not v[2]:
            other_rx_failures+=1
            if len(v)>=6 and int(v[4])>=0:
                direction_other_phy[0 if int(v[1])==0 else 1,int(v[4])]+=1
    accounting={}
    if records:
        lengths={len(v) for v in records}
        if lengths not in ({16},{17},{19}):
            raise ValueError("Mixed or unsupported STEP packet accounting schema")
        accounting_version=4 if lengths=={19} else (3 if lengths=={17} else 2)
        by_key={}; coverage={}
        for v in records:
            if any(x<0 or x!=int(x) for x in v[1:]):
                raise ValueError("Invalid STEP packet accounting record")
            key=(int(v[1]),int(v[2])); at_time=coverage.setdefault(v[0],set())
            if key in at_time: raise ValueError("Duplicate STEP packet accounting")
            at_time.add(key)
            if v[9]+v[3]!=v[4]+sum(v[5:9])+v[10]:
                raise ValueError("Packet conservation failed")
            if key not in by_key: by_key[key]=dict(start=v[9],end=v[9],totals=np.zeros(len(v)))
            item=by_key[key]
            if item['end']!=v[9]: raise ValueError("Discontinuous pending packet counts")
            item['end']=v[10]; item['totals']+=v
        expected={(node,ac) for node in range(num_sta+1) for ac in range(4)}
        if any(keys!=expected for keys in coverage.values()):
            raise ValueError("Incomplete STEP packet accounting")
        sums=np.sum([x['totals'] for x in by_key.values()],axis=0)
        offered_packets=sums[3]; failed=sums[7]; discarded=sums[5]+sums[6]+sums[8]
        retry=sums[13]; attempts=sums[14]; timeouts=sums[15]
        accounting['accounting_version']=accounting_version
        for ac,name in enumerate(('vo','vi','be','bk')):
            items=[x for k,x in by_key.items() if k[1]==ac]
            v=np.sum([x['totals'] for x in items],axis=0); offered[ac]=v[11]
            if len(delays[ac])!=v[4] or ac_bytes[ac]!=v[12]:
                raise ValueError("ACK samples disagree with packet accounting")
            accounting.update({f'{name}_offered_packets':int(v[3]),f'{name}_delivered_packets':int(v[4]),
                f'{name}_overflow_count':int(v[5]),f'{name}_expired_count':int(v[6]),
                f'{name}_failed_count':int(v[7]),f'{name}_other_drop_count':int(v[8]),
                f'{name}_pending_packets_start':int(sum(x['start'] for x in items)),
                f'{name}_pending_packets_end':int(sum(x['end'] for x in items)),
                f'{name}_drop_rate':float(sum(v[5:9])/max(v[3],1))})
            at_risk=int(sum(x['start'] for x in items))+int(v[3])
            final_failures=int(sum(v[5:9]))
            accounting.update({f'{name}_at_risk_packets':at_risk,
                f'{name}_delivery_fraction':float(v[4]/max(at_risk,1)),
                f'{name}_final_failure_fraction':float(final_failures/max(at_risk,1)),
                f'{name}_unfinished_fraction':float(sum(x['end'] for x in items)/max(at_risk,1)),
                f'{name}_queue_rejection_fraction_of_generated':float(v[5]/max(v[3],1)),
                f'{name}_attempts':int(v[14]),f'{name}_timeouts':int(v[15])})
            if accounting_version>=3:
                accounting[f'{name}_internal_collision_count']=int(v[16])
            if accounting_version>=4:
                eligible=int(v[17]); on_time=int(v[18])
                if on_time>eligible: raise ValueError('On-time ACK count exceeds deadline-eligible packets')
                accounting.update({f'{name}_deadline_eligible_packets':eligible,
                    f'{name}_on_time_delivered_packets':on_time,
                    f'{name}_on_time_delivery_rate':float(on_time/max(eligible,1)),
                    f'{name}_on_time_delivery_valid':bool(eligible)})
        accounting.update(delivered_packets=int(sums[4]),overflow_count=int(sums[5]),
            expired_count=int(sums[6]),other_drop_count=int(sums[8]),
            pending_packets_start=int(sum(x['start'] for x in by_key.values())),
            pending_packets_end=int(sum(x['end'] for x in by_key.values())))
        at_risk=int(sum(x['start'] for x in by_key.values()))+int(sums[3])
        final_failures=int(sum(sums[5:9]))
        accounting.update(at_risk_packets=at_risk,
            delivery_fraction=float(sums[4]/max(at_risk,1)),
            final_failure_fraction=float(final_failures/max(at_risk,1)),
            unfinished_fraction=float(sum(x['end'] for x in by_key.values())/max(at_risk,1)),
            queue_rejection_fraction_of_generated=float(sums[5]/max(sums[3],1)))
        if accounting_version>=3:
            accounting['internal_collision_count']=int(sums[16])
        if accounting_version>=4:
            accounting.update(deadline_eligible_packets=int(sums[17]),on_time_delivered_packets=int(sums[18]))
    total=sum(sta_bytes)
    internal_collisions=accounting.get('internal_collision_count',0)
    result={"goodput_mbps":float(sum(ac_bytes)*8/1e6/duration),
            "ack_cts_timeout_ratio":float(timeouts/max(attempts,1)),
            "retry_count":retry,"failed_count":failed,"discarded_count":discarded,
            "attempts":attempts,"timeouts":timeouts,
            "collision_rate":collisions/max(air_attempts,1),"collision_count":collisions,
            "external_collision_rate":collisions/max(air_attempts,1),
            "internal_collision_count":internal_collisions,
            "internal_collision_per_access_event":internal_collisions/max(air_attempts+internal_collisions,1),
            "combined_collision_event_fraction":(collisions+internal_collisions)/max(air_attempts+internal_collisions,1),
            "air_attempts":air_attempts,"other_rx_failures":other_rx_failures,
            "offered_packets":offered_packets,"drop_rate":(failed+discarded)/max(offered_packets,1),
            "jain_fairness":float(total**2/(num_sta*sum(sta_bytes**2))) if total>0 else 1.,
            "report_rate_per_sta":reports/num_sta/duration}
    for c,ac in enumerate(("vo","vi","be","bk")):
        result.update({f"{ac}_goodput_mbps":float(ac_bytes[c]*8/1e6/duration),
                       f"{ac}_offered_mbps":float(offered[c]*8/1e6/duration),
                       f"{ac}_delay_mean_ms":float(np.mean(delays[c])) if delays[c] else 0.,
                       f"{ac}_delay_p95_ms":float(np.percentile(delays[c],95)) if delays[c] else 0.,
                       f"{ac}_delay_valid":bool(delays[c])})
        for direction_index,direction in enumerate(("ul","dl")):
            samples=direction_delays[direction_index][c]
            result.update({f"{direction}_{ac}_goodput_mbps":float(direction_bytes[direction_index,c]*8/1e6/duration),
                           f"{direction}_{ac}_delay_mean_ms":float(np.mean(samples)) if samples else 0.,
                           f"{direction}_{ac}_delay_p95_ms":float(np.percentile(samples,95)) if samples else 0.,
                           f"{direction}_{ac}_delay_valid":bool(samples),
                           f"{direction}_{ac}_external_collision_count":int(direction_collisions[direction_index,c]),
                           f"{direction}_{ac}_other_phy_failure_count":int(direction_other_phy[direction_index,c])})
    result.update(accounting)
    return result


def aggregate_accounting(rows: list[dict]) -> dict:
    """Episode counts are sums; queue endpoints are first/last, never means."""
    if not rows or all('accounting_version' not in r for r in rows): return {}
    versions={r.get('accounting_version') for r in rows}
    if len(versions)!=1 or next(iter(versions)) not in (2,3,4):
        raise ValueError('Mixed accounting schemas')
    result={'accounting_version':next(iter(versions))}
    for key in rows[0]:
        if key.endswith(('_count','_packets','_attempts','_timeouts')) or key in ('attempts','timeouts','air_attempts','other_rx_failures'):
            result[key]=int(sum(r[key] for r in rows))
        elif key.endswith('pending_packets_start'): result[key]=int(rows[0][key])
        elif key.endswith('pending_packets_end'): result[key]=int(rows[-1][key])
    for prefix in ('','vo_','vi_','be_','bk_'):
        if any(a[prefix+'pending_packets_end']!=b[prefix+'pending_packets_start'] for a,b in zip(rows,rows[1:])):
            raise ValueError('Discontinuous episode pending packet counts')
        dropped=sum(result[prefix+k] for k in ('overflow_count','expired_count','failed_count','other_drop_count'))
        at_risk=result[prefix+'pending_packets_start']+result[prefix+'offered_packets']
        if at_risk!=result[prefix+'delivered_packets']+dropped+result[prefix+'pending_packets_end']:
            raise ValueError('Episode packet conservation failed')
        result[prefix+'drop_rate']=dropped/max(result[prefix+'offered_packets'],1)
        result[prefix+'accounted_loss_fraction']=dropped/max(at_risk,1)
        result[prefix+'pending_fraction_end']=result[prefix+'pending_packets_end']/max(at_risk,1)
        result[prefix+'at_risk_packets']=at_risk
        result[prefix+'delivery_fraction']=result[prefix+'delivered_packets']/max(at_risk,1)
        result[prefix+'final_failure_fraction']=dropped/max(at_risk,1)
        result[prefix+'unfinished_fraction']=result[prefix+'pending_packets_end']/max(at_risk,1)
        result[prefix+'queue_rejection_fraction_of_generated']=result[prefix+'overflow_count']/max(result[prefix+'offered_packets'],1)
        if next(iter(versions))>=4 and prefix in ('vo_','vi_'):
            eligible=result[prefix+'deadline_eligible_packets']; on_time=result[prefix+'on_time_delivered_packets']
            result[prefix+'on_time_delivery_rate']=on_time/max(eligible,1)
            result[prefix+'on_time_delivery_valid']=bool(eligible)
    return result
