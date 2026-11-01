"""Summarize isolated baseline probes; delay plots are strict successful-packet CCDFs."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT
from scripts.audit_baseline import ACS, analyze_trace


def main():
    p=argparse.ArgumentParser(); p.add_argument('run',type=Path); args=p.parse_args()
    root=args.run; out=ROOT/'results/figures'/root.name; out.mkdir(parents=True,exist_ok=True)
    records=[]; data={}; queues={}
    for folder in sorted(root.glob('S*_seed101')):
        meta=json.loads((folder/'metadata.json').read_text())
        # Rebuild the summary with the current cohort/deadline checks, preserving raw traces.
        report=analyze_trace(folder/'mac_trace.csv',meta['duration_s'],folder/'analysis')
        data[folder.name]=np.load(folder/'analysis/delay_samples.npz')
        queues[folder.name]=list(csv.DictReader((folder/'analysis/queue_timeline.csv').open()))
        for ac in ACS:
            rows=[r for r in report['rows'] if r['ac']==ac]
            samples=np.concatenate([data[folder.name][f'{ac}_{d}'] for d in ('DL','UL')])
            r=dict(case=folder.name,ac=ac)
            for key in ('offered','success','unreported_overflow','reported_overflow','expired','retry_limit','qos_old','pending','offered_mbps','goodput_mbps'):
                r[key]=sum(x[key] for x in rows)
            r['drop_fraction']=(r['offered']-r['success']-r['pending'])/r['offered']
            r['previously_reported_drop_fraction']=(r['expired']+r['retry_limit']+r['qos_old']+r['reported_overflow'])/r['offered']
            for q in (50,95,99,99.9,100): r[f'p{q}_ms']=float(np.percentile(samples[:,1],q))
            r['departure_after10_p95_ms']=float(np.percentile(samples[samples[:,0]>=10,1],95))
            # A period-p95 average is exported explicitly, never mislabeled as pooled p95.
            bins=np.floor(samples[:,0]*10).astype(int)
            r['after10_mean_100ms_window_p95_ms']=float(np.mean([np.percentile(samples[bins==b,1],95) for b in np.unique(bins[bins>=100])]))
            for d in (20,50):
                r[f'success_ccdf_{d}ms']=float(np.mean(samples[:,1]>d))
                r[f'eligible_{d}ms']=sum(x[f'eligible_{d}ms'] for x in rows)
                r[f'on_time_{d}ms']=sum(x[f'on_time_{d}ms'] for x in rows)
                r[f'not_delivered_by_{d}ms_fraction']=1-r[f'on_time_{d}ms']/r[f'eligible_{d}ms']
            records.append(r)
    with (root/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    for scenario in ('S2_realtime_stress','S4_overload'):
        fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
        names=[n for n in data if n.startswith(scenario)]
        for ax,ac in zip(axes.flat,ACS):
            for name in names:
                d=np.sort(np.concatenate([data[name][f'{ac}_{di}'][:,1] for di in ('DL','UL')]))
                # Retain the deepest observed tail instead of truncating at the
                # quantile-grid resolution (roughly 1/2000).
                x=np.unique(np.r_[np.quantile(d,np.linspace(0,1,min(len(d),2000))),d[-50:]])
                y=(len(d)-np.searchsorted(d,x,side='right'))/len(d)
                ttl=10 if '_ttl10_' in name else 1000
                retry=255 if 'retry255' in name else 7
                label=f'TTL {ttl} s, retry {retry}'
                ax.step(x[y>0],y[y>0],where='post',label=label)
            ax.set(xscale='log',yscale='log',title=ac.upper(),xlabel='MAC delay (ms)',ylabel='P(D > x | ACKed)')
            ax.grid(alpha=.25); ax.legend(fontsize=8)
        fig.suptitle(f'{scenario}: baseline, 60 s, seed 101\nTTL sensitivity; conditional on successful ACK')
        for ext in ('png','pdf'): fig.savefig(out/f'{scenario}_ccdf.{ext}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for i,scenario in enumerate(('S2_realtime_stress','S4_overload')):
        for j,ac in enumerate(('be','bk')):
            ax=axes[i,j]
            for name,rows in queues.items():
                if not name.startswith(scenario) or 'retry255' in name: continue
                dl=[r for r in rows if r['ac']==ac and r['node']=='0']
                ax.plot([float(r['time_s']) for r in dl],[int(r['packets']) for r in dl],
                        label=name.removeprefix(scenario+'_').removesuffix('_seed101'),alpha=.8)
            ax.axhline(500,color='black',linestyle=':',label='queue limit')
            ax.set(title=f'{scenario[:2]} AP {ac.upper()}',xlabel='Time (s)',ylabel='Queued packets')
            ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle('Default EDCA: AP backlog; queue limit = 500 packets per AC; seed 101')
    for ext in ('png','pdf'): fig.savefig(out/f'baseline_queue_backlog.{ext}',dpi=180)
    plt.close(fig)
    lines=['# Baseline 逐包诊断结果','',
           '单 seed 101、60 秒；诊断参数不可作为正式方法收益。时延仅统计成功包，丢弃与期末剩余另列。', '',
           '| case | AC | p95 ms | 溢出漏记 | 过期 | 重试丢弃 | 剩余 | 总丢弃率 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['case']} | {r['ac']} | {r['p95_ms']:.3f} | {r['unreported_overflow']} | {r['expired']} | {r['retry_limit']} | {r['pending']} | {r['drop_fraction']:.2%} |")
    (root/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(out)


if __name__=='__main__': main()
