"""Isolated packet-accounting/TTL probes; never replace the production executable.

Build an observer-only copy of wifi-control.cc. Extra CSV events expose enqueue
acceptance and queue occupancy without changing traffic, MAC rules or S counters.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, shutil, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config
from sim.backend.ns3_backend import Ns3Backend, runtime_env, find_binary
from sim.backend.base_backend import ControlAction

ACS = ('vo', 'vi', 'be', 'bk')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instrument(source):
    if "Accounting schema 2" in source:
        # Production now supplies the same optional UID/queue trace itself.
        return source
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f'Observer insertion anchor changed: {old[:80]}')
        source = source.replace(old, new)
    replace('    void Drop(uint32_t node,', '''    uint64_t auditSendingUid=0;
    bool auditEnqueued=false, auditDropped=false;
    void AuditEnqueue(Ptr<const WifiMpdu> mpdu)
    {
        if(mpdu->GetPacket()->GetUid()==auditSendingUid) auditEnqueued=true;
    }

    void Drop(uint32_t node,''')
    replace('        auto& c=counters[node][tag.ac];\n        if(reason',
            '        if(mpdu->GetPacket()->GetUid()==auditSendingUid) auditDropped=true;\n'
            '        auto& c=counters[node][tag.ac];\n        if(reason')
    replace('        devices[node]->Send(packet,dest,0x88b5);', '''        auditSendingUid=packet->GetUid(); auditEnqueued=false; auditDropped=false;
        Trace(node,"OFFERED",ac,bytes,auditSendingUid);
        devices[node]->Send(packet,dest,0x88b5);
        if(!auditEnqueued && !auditDropped)
            Trace(node,"UNREPORTED_ENQUEUE_DROP",ac,0,auditSendingUid);''')
    replace('                q->GetWifiMacQueue()->SetMaxDelay(Seconds(Num("queue_delay_s")));',
            '                q->GetWifiMacQueue()->SetMaxDelay(Seconds(Num("queue_delay_s")));\n'
            '                q->GetWifiMacQueue()->TraceConnectWithoutContext("Enqueue",MakeCallback(&Experiment::AuditEnqueue,this));')
    replace('                queue->WipeAllExpiredMpdus();', '''                queue->WipeAllExpiredMpdus();
                Trace(node,"QUEUE_PACKETS",ac,queue->QueueBase::GetNPackets());
                if(Now()<sample+1e-6) {
                    auto q=devices[node]->GetMac()->GetQosTxop(acs[ac]);
                    Trace(node,"QUEUE_LIMIT",ac,queue->GetMaxSize().GetValue());
                    Trace(node,"QUEUE_TTL",ac,queue->GetMaxDelay().GetSeconds());
                    Trace(node,"ACTUAL_CWMIN",ac,q->GetMinCw(0));
                    Trace(node,"ACTUAL_CWMAX",ac,q->GetMaxCw(0));
                    Trace(node,"ACTUAL_AIFSN",ac,q->GetAifsn(0));
                    Trace(node,"ACTUAL_TXOP_US",ac,q->GetTxopLimit().GetMicroSeconds());
                }''')
    return source


def build_observer(root):
    original = ROOT/'ns3_ext/wifi-control.cc'
    source = instrument(original.read_text(encoding='utf-8'))
    scratch = ROOT/'vendor/ns-3.44/scratch/wifi-baseline-audit.cc'
    scratch.write_text(source, encoding='utf-8')
    (root/'wifi-baseline-audit.cc').write_text(source, encoding='utf-8')
    cmake = shutil.which('cmake', path=runtime_env()['PATH'])
    with (root/'build.log').open('w') as out:
        subprocess.run([cmake, '--build', str(ROOT/'vendor/ns-3.44/cmake-cache'),
                        '--parallel', '4', '--target', 'scratch_wifi-baseline-audit'],
                       env=runtime_env(), stdout=out, stderr=subprocess.STDOUT, check=True)
    binaries = list((ROOT/'vendor/ns-3.44/build/scratch').glob('ns3.44-wifi-baseline-audit-*.exe'))
    if len(binaries)!=1:
        raise FileNotFoundError(f'Expected one observer binary: {binaries}')
    return binaries[0]


def analyze_trace(path, duration, out):
    """Every offered UID has exactly one terminal outcome or is right-censored."""
    pending = {}; counts = defaultdict(Counter); delays = defaultdict(list)
    arrivals = defaultdict(int); delivered_bytes = Counter(); arrival_bytes = Counter()
    queue_rows=[]; settings=[]; outcomes=Counter(); overflow_ages=[]
    eligible=Counter(); on_time=Counter()
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            event=row['event']; ac=int(row['ac']); node=int(row['node'])
            if ac < 0: continue
            key=(ac, 'DL' if node==0 else 'UL'); uid=int(row['uid']); t=float(row['time_s'])
            if event=='OFFERED':
                if uid in pending or uid in outcomes: raise ValueError('Duplicate offered UID')
                pending[uid]=(key,t,int(float(row['value'])),node)
                counts[key]['offered']+=1; arrival_bytes[key]+=int(float(row['value']))
                arrivals[(int(t),ac,key[1])]+=1
                for deadline in (20,50):
                    if t <= duration-deadline/1000: eligible[(key,deadline)]+=1
            elif event in ('ACKED','DROP','UNREPORTED_ENQUEUE_DROP'):
                if uid not in pending: raise ValueError(f'Unmatched outcome {event} {uid}')
                k,created,size,_=pending.pop(uid)
                if k!=key: raise ValueError('AC/direction mismatch')
                outcomes[uid]+=1
                if event=='ACKED':
                    counts[key]['success']+=1; delivered_bytes[key]+=size
                    delay=float(row['value'])
                    if abs(delay-(t-created)*1000)>0.001: raise ValueError('Delay timestamp mismatch')
                    delays[key].append((t,delay))
                    for deadline in (20,50):
                        if created <= duration-deadline/1000 and delay <= deadline:
                            on_time[(key,deadline)]+=1
                else:
                    reason='unreported_overflow' if event=='UNREPORTED_ENQUEUE_DROP' else {
                        0:'reported_overflow',1:'expired',2:'retry_limit',3:'qos_old'}[int(float(row['value']))]
                    counts[key][reason]+=1
                    if reason=='unreported_overflow': overflow_ages.append(t-created)
            elif event=='QUEUE_PACKETS':
                queue_rows.append(dict(time_s=t,node=node,ac=ACS[ac],packets=int(float(row['value']))))
            elif event.startswith('ACTUAL_') or event in ('QUEUE_LIMIT','QUEUE_TTL'):
                settings.append(dict(node=node,ac=ACS[ac],setting=event,value=float(row['value'])))
    pending_ages=defaultdict(list); pending_by_node=Counter()
    for key,created,size,node in pending.values():
        counts[key]['pending']+=1; pending_ages[key].append((duration-created)*1000)
        pending_by_node[(node,ACS[key[0]])]+=1
    latest={}
    for row in queue_rows: latest[(row['node'],row['ac'])]=row['packets']
    mismatch={str(k):[latest.get(k,0),pending_by_node[k]] for k in set(latest)|set(pending_by_node)
              if latest.get(k,0)!=pending_by_node[k]}
    if mismatch: raise ValueError(f'Queue conservation failed: {mismatch}')
    result=[]
    for key,c in sorted(counts.items()):
        samples=np.asarray([d for _,d in delays[key]])
        dropped=sum(c[k] for k in ('unreported_overflow','reported_overflow','expired','retry_limit','qos_old'))
        if c['offered']!=c['success']+dropped+c['pending']: raise ValueError('Packet conservation failed')
        r=dict(ac=ACS[key[0]],direction=key[1],**{k:c[k] for k in
               ('offered','success','unreported_overflow','reported_overflow','expired','retry_limit','qos_old','pending')},
               offered_mbps=arrival_bytes[key]*8/1e6/duration,goodput_mbps=delivered_bytes[key]*8/1e6/duration,
               true_drop_fraction=dropped/c['offered'],
               pending_age_max_ms=max(pending_ages[key],default=0))
        for q in (50,95,99,99.9,100): r[f'delay_p{q}_ms']=float(np.percentile(samples,q)) if len(samples) else None
        after_warmup=[d for t,d in delays[key] if t>=10]
        r['departure_10s_to_end_p95_ms']=float(np.percentile(after_warmup,95)) if after_warmup else None
        for threshold in (20,50,10000):
            r[f'success_ccdf_{threshold}ms']=float(np.mean(samples>threshold)) if len(samples) else None
        for deadline in (20,50):
            n=eligible[(key,deadline)]
            r[f'eligible_{deadline}ms']=n
            r[f'on_time_{deadline}ms']=on_time[(key,deadline)]
            r[f'not_delivered_by_{deadline}ms_fraction']=1-on_time[(key,deadline)]/n if n else None
        result.append(r)
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('accounting.csv',result),('queue_timeline.csv',queue_rows),('actual_settings.csv',settings),
                      ('arrival_bins.csv',[dict(second=k[0],ac=ACS[k[1]],direction=k[2],packets=v) for k,v in sorted(arrivals.items())])]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez_compressed(out/'delay_samples.npz',**{f'{ACS[k[0]]}_{k[1]}':np.asarray(v).reshape(-1,2) for k,v in delays.items()})
    report=dict(duration_s=duration,rows=result,conservation_passed=True,
                final_pending_matches_queue=True,max_unreported_overflow_age_s=max(overflow_ages,default=0))
    (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def run_probe(config, scenario, seed, duration, folder):
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    started=time.monotonic()
    with Ns3Backend(config,scenario,seed,folder,trace=True) as backend:
        for _ in range(round(duration)):
            backend.step(ControlAction(),1.)
        # Legacy protocol did not drain deferred expiry at the final boundary.
        if backend.accounting_version==1:
            backend.step(ControlAction(),1e-8)
        duration=backend.time
    report=analyze_trace(folder/'mac_trace.csv',duration,folder/'analysis')
    (folder/'metadata.json').write_text(json.dumps(dict(seed=seed,scenario=scenario,duration_s=duration,
        runtime_s=time.monotonic()-started,binary_sha256=sha(find_binary(config))),indent=2),encoding='utf-8')
    print(json.dumps(dict(case=folder.name,conservation=report['conservation_passed'],runtime_s=time.monotonic()-started)),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); root=args.output.resolve(); root.mkdir(parents=True,exist_ok=False)
    cfg=load_config(ROOT/'configs/full.yaml'); production=find_binary(cfg)
    before=dict(source=sha(ROOT/'ns3_ext/wifi-control.cc'),binary=sha(production))
    binary=build_observer(root); audit=copy.deepcopy(cfg); audit['backend']['binary']=str(binary)
    cap=json.loads((ROOT/cfg['traffic']['capacity_file']).read_text())['C_ref_mbps']
    scenarios=load_config(ROOT/'configs/scenarios_legacy_rho.yaml')
    s=dict(scenarios['S2_realtime_stress'],offered_mbps=cap*scenarios['S2_realtime_stress']['rho'])
    # Observer equivalence: same seeds, command boundaries and production stdout.
    for name,c in [('production_check',cfg),('observer_check',audit)]:
        d=root/name
        with Ns3Backend(c,s,101,d,trace=True) as backend:
            for _ in range(2): backend.step(ControlAction(),1.)
    equal=(root/'production_check/events.log').read_bytes()==(root/'observer_check/events.log').read_bytes()
    if not equal: raise RuntimeError('Observer changed simulation stdout')
    (root/'observer_validation.json').write_text(json.dumps(dict(stdout_identical=equal,
        original_hashes=before,observer_sha256=sha(binary)),indent=2),encoding='utf-8')
    jobs=[]
    for name in ('S2_realtime_stress','S4_overload'):
        s=dict(scenarios[name],offered_mbps=cap*scenarios[name]['rho'])
        for ttl in (10,1000):
            c=copy.deepcopy(audit); c['queues']['max_delay_s']=ttl
            jobs.append((c,s,101,60,root/f'{name}_ttl{ttl}_seed101'))
    # Diagnostic sensitivity only: not a proposed production retry setting.
    name='S2_realtime_stress'; s=dict(scenarios[name],offered_mbps=cap*scenarios[name]['rho'])
    c=copy.deepcopy(audit); c['queues']['max_delay_s']=1000; c['network']['retry_limit']=255
    jobs.append((c,s,101,60,root/f'{name}_ttl1000_retry255_seed101'))
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_probe,*job) for job in jobs]
        for future in futures: future.result()
    after=dict(source=sha(ROOT/'ns3_ext/wifi-control.cc'),binary=sha(production))
    if before!=after: raise RuntimeError('Production artifact changed')
    (root/'complete.json').write_text(json.dumps(dict(production_unchanged=True,original_hashes=after),indent=2))


if __name__=='__main__': main()
