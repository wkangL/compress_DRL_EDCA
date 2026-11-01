"""Replay preserved 60 s baseline probes against corrected production accounting."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT,load_config,backend_source_hash
from scripts.audit_baseline import analyze_trace,sha
from sim.backend.base_backend import ControlAction
from sim.backend.ns3_backend import Ns3Backend,find_binary
from sim.statistics.metrics import summarize,aggregate_accounting
from rl.trainer import save_csv


def selected_events(path):
    digest=hashlib.sha256(); count=0
    with Path(path).open('rb') as stream:
        for line in stream:
            # D includes ACK delays; A/C/F are attempted frames and PHY failures.
            if line[:2] in (b'D ',b'A ',b'C ',b'F '):
                digest.update(line.replace(b'\r\n',b'\n')); count+=1
    return dict(sha256=digest.hexdigest(),records=count)


def run_case(args):
    name,root,legacy=args; started=time.monotonic()
    config=load_config(ROOT/'configs/full.yaml')
    # Preserve the precise historical workload independently of new calibration.
    definition=load_config(ROOT/'configs/scenarios_legacy_rho.yaml')[name]
    scenario=dict(definition,offered_mbps=definition['rho']*27.30225)
    folder=Path(root)/name; folder.mkdir()
    periods=[]; delays=[[] for _ in range(4)]
    with Ns3Backend(config,scenario,101,folder,trace=True) as backend:
        for _ in range(60):
            events=backend.step(ControlAction(),1.)
            periods.append(dict(time_s=backend.time,**summarize(events,10,1.)))
            for e in events:
                if e.kind=='D': delays[int(e.values[2])].append(e.values[4])
        if backend.time!=60.: raise AssertionError('STEP advanced past requested time')
    totals=aggregate_accounting(periods)
    for ac,samples in zip(('vo','vi','be','bk'),delays):
        totals[ac+'_delay_p95_ms']=float(np.percentile(samples,95)) if samples else 0.
    totals['goodput_mbps']=sum(m['goodput_mbps'] for m in periods)/60
    audit=analyze_trace(folder/'mac_trace.csv',60.,folder/'analysis')
    reference=Path(legacy)/f'{name}_ttl10_seed101'
    old=json.loads((reference/'analysis/audit.json').read_text())
    fields={'offered_packets':['offered'],'delivered_packets':['success'],
            'overflow_count':['reported_overflow','unreported_overflow'],
            'expired_count':['expired'],'failed_count':['retry_limit'],
            'other_drop_count':['qos_old'],'pending_packets_end':['pending']}
    for prefix in ('','vo_','vi_','be_','bk_'):
        for key,columns in fields.items():
            for report in (old,audit):
                expected=sum(row[col] for row in report['rows'] for col in columns
                             if not prefix or row['ac']==prefix[:-1])
                if totals[prefix+key]!=expected:
                    raise AssertionError(f'{name} {prefix+key}: {totals[prefix+key]} != {expected}')
    if any(row['unreported_overflow'] for row in audit['rows']):
        raise AssertionError('Unreported drop remains')
    before=selected_events(reference/'events.log'); after=selected_events(folder/'events.log')
    if before!=after: raise AssertionError(f'{name} MAC/PHY trajectory changed: {before} / {after}')
    save_csv(folder/'periods.csv',periods)
    result=dict(scenario=name,seed=101,duration_s=60.,**totals,uid_audit_passed=True,
                packet_conservation_passed=True,legacy_events_identical=True,events=after,
                source_hash=backend_source_hash(),binary_sha256=sha(find_binary(config)),
                runtime_s=time.monotonic()-started)
    (folder/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (folder/'config.json').write_text(json.dumps(dict(config=config,scenario=scenario),indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--legacy',type=Path,default=ROOT/'results/raw/baseline_audit_20261019_v4')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    jobs=[(name,args.output,args.legacy) for name in ('S2_realtime_stress','S4_overload')]
    with ProcessPoolExecutor(max_workers=2) as pool: results=list(pool.map(run_case,jobs))
    save_csv(args.output/'comparison.csv',results)
    (args.output/'complete.json').write_text(json.dumps(dict(passed=True,cases=results),indent=2),encoding='utf-8')


if __name__=='__main__': main()
