"""Packet accounting must cover real queue rejection and deferred expiry."""
import csv
import os
from pathlib import Path
import tempfile
import unittest

from project_config import ROOT, load_config
from sim.backend.base_backend import ControlAction, Event
from sim.backend.ns3_backend import Ns3Backend
from sim.statistics.metrics import summarize, aggregate_accounting


def records(time=1., start=0, offered=10, success=2, overflow=3, expired=1, retry=1, other=0,
            internal=None,deadline=None):
    end=start+offered-success-overflow-expired-retry-other
    events=[]
    for node in range(2):
        for ac in range(4):
            values=(offered,success,overflow,expired,retry,other,start,end,offered*1500,success*1500,4,7,5) if (node,ac)==(1,2) else (0,)*13
            if internal is not None: values=(*values,internal if (node,ac)==(1,2) else 0)
            if deadline is not None:
                eligible,on_time=deadline
                values=(*values,eligible if (node,ac)==(1,2) else 0,on_time if (node,ac)==(1,2) else 0)
            events.append(Event('P',(time,node,ac,*values)))
    events += [Event('D',(time,0,2,1500,3,1)) for _ in range(success)]
    return events


class AccountingTests(unittest.TestCase):
    def test_categories_and_sample_not_double_counted(self):
        events=records()+[Event('S',(1,1,2,0,0,999,999,999,1,999,999,999,999,999))]
        m=summarize(events,1,1.)
        self.assertEqual((m['offered_packets'],m['delivered_packets'],m['pending_packets_end']),(10,2,3))
        self.assertEqual((m['overflow_count'],m['expired_count'],m['failed_count']),(3,1,1))
        self.assertEqual(m['drop_rate'],.5)

    def test_incomplete_duplicate_and_unbalanced_rejected(self):
        events=records()
        for bad in (events[1:],events+[events[0]],records(offered=-1)):
            with self.assertRaises(ValueError): summarize(bad,1,1.)
        bad=events.copy(); v=list(bad[6].values); v[10]+=1; bad[6]=Event('P',tuple(v))
        with self.assertRaisesRegex(ValueError,'conservation'): summarize(bad,1,1.)

    def test_ack_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'ACK samples'): summarize(records()[:-1],1,1.)

    def test_episode_totals_and_pending_continuity(self):
        rows=[summarize(records(),1,1.),summarize(records(time=2,start=3),1,1.)]
        total=aggregate_accounting(rows)
        self.assertEqual((total['offered_packets'],total['overflow_count'],total['pending_packets_end']),(20,6,6))
        self.assertEqual(total['drop_rate'],.5)
        # Individually balanced periods still cannot hide discontinuity.
        rows[1]=summarize(records(time=2,start=4),1,1.)
        with self.assertRaisesRegex(ValueError,'Discontinuous'): aggregate_accounting(rows)
        with self.assertRaisesRegex(ValueError,'Mixed'): aggregate_accounting([{},rows[0]])

    def test_schema3_internal_collision_count(self):
        first=summarize(records(internal=9),1,1.)
        second=summarize(records(time=2,start=3,internal=4),1,1.)
        self.assertEqual(first['accounting_version'],3)
        self.assertEqual(first['internal_collision_count'],9)
        self.assertEqual(first['be_internal_collision_count'],9)
        self.assertEqual(aggregate_accounting([first,second])['internal_collision_count'],13)
        with self.assertRaisesRegex(ValueError,'Mixed'):
            aggregate_accounting([first,summarize(records(),1,1.)])

    def test_schema4_causal_deadline_counts(self):
        first=summarize(records(internal=2,deadline=(7,5)),1,1.)
        second=summarize(records(time=2,start=3,internal=1,deadline=(4,3)),1,1.)
        self.assertEqual(first['accounting_version'],4)
        self.assertEqual(first['be_deadline_eligible_packets'],7)
        self.assertAlmostEqual(first['be_on_time_delivery_rate'],5/7)
        total=aggregate_accounting([first,second])
        # BE/BK deadlines are disabled in production, but synthetic records
        # verify that packet counters survive schema parsing and aggregation.
        self.assertEqual(total['be_deadline_eligible_packets'],11)
        bad=records(internal=0,deadline=(2,3))
        with self.assertRaisesRegex(ValueError,'On-time ACK'):
            summarize(bad,1,1.)


@unittest.skipUnless(os.environ.get('WIFI_INTEGRATION')=='1','real ns-3 integration is opt-in')
class NativeAccountingTests(unittest.TestCase):
    def run_queue_case(self, ttl, limit, periods):
        c=load_config(ROOT/'configs/quick.yaml'); c['network']['num_sta']=2
        c['queues'].update(max_delay_s=ttl,limits_packets=[limit]*4)
        scenario=dict(offered_mbps=100.,ac_mix=[0,0,1,0],ul_fraction=0.)
        with tempfile.TemporaryDirectory(dir=ROOT/'results/raw') as tmp:
            directory=Path(tmp); rows=[]; sample_drops=0
            with Ns3Backend(c,scenario,101,directory,trace=True) as backend:
                for duration in periods:
                    events=backend.step(ControlAction(),duration)
                    rows.append(summarize(events,2,duration))
                    sample_drops+=sum(e.values[6]+e.values[7] for e in events if e.kind=='S')
                self.assertAlmostEqual(backend.time,sum(periods),places=9)
            total=aggregate_accounting(rows)
            with (directory/'mac_trace.csv').open() as stream:
                trace=list(csv.DictReader(stream))
            drops=[r for r in trace if r['event']=='DROP']
            self.assertEqual(len({r['uid'] for r in drops}),len(drops))
            for reason,key in [(0,'overflow_count'),(1,'expired_count'),(2,'failed_count')]:
                self.assertEqual(sum(float(r['value'])==reason for r in drops),total[key])
            self.assertEqual(sum(r['event']=='OFFERED' for r in trace),total['offered_packets'])
            return total,sample_drops

    def test_native_overflow_trace(self):
        total,samples=self.run_queue_case(10.,2,[.1,.1,.1])
        self.assertGreater(total['overflow_count'],0)
        self.assertEqual(samples,total['discarded_count']+total['failed_count'])

    def test_expiry_at_final_sample(self):
        # Short lifetime is a test fixture, never the production setting.
        total,samples=self.run_queue_case(.02,10000,[.1,.1,.1])
        self.assertGreater(total['expired_count'],0)
        self.assertEqual(samples,total['discarded_count']+total['failed_count'])

    def test_partial_sample_interval(self):
        total,samples=self.run_queue_case(10.,2,[.15,.11])
        self.assertGreater(total['overflow_count'],0)
        self.assertLess(samples,total['discarded_count']+total['failed_count'])

    def test_internal_edcaf_collision_trace(self):
        c=load_config(ROOT/'configs/quick.yaml'); c['network']['num_sta']=2
        scenario=dict(offered_mbps=80.,ac_mix=[.25,.25,.25,.25],ul_fraction=.5)
        with tempfile.TemporaryDirectory(dir=ROOT/'results/raw') as tmp:
            directory=Path(tmp); rows=[]
            with Ns3Backend(c,scenario,103,directory,trace=True) as backend:
                for _ in range(2): rows.append(summarize(backend.step(ControlAction(),1.),2,1.))
                self.assertEqual(backend.accounting_version,4)
            total=aggregate_accounting(rows)
            with (directory/'mac_trace.csv').open() as stream: trace=list(csv.DictReader(stream))
            observed=sum(r['event']=='INTERNAL_COLLISION' for r in trace)
            self.assertGreater(observed,0)
            self.assertEqual(observed,total['internal_collision_count'])

    def test_causal_deadline_ledger_matches_native_trace(self):
        """Schema-4 verdict counts must equal packet-level causal trace verdicts."""
        c=load_config(ROOT/'configs/quick.yaml'); c['network']['num_sta']=2
        scenario=dict(offered_mbps=80.,ac_mix=[.5,.5,0,0],ul_fraction=.5)
        horizon=2.
        with tempfile.TemporaryDirectory(dir=ROOT/'results/raw') as tmp:
            directory=Path(tmp)
            with Ns3Backend(c,scenario,107,directory,trace=True) as backend:
                total=aggregate_accounting([summarize(
                    backend.step(ControlAction(),horizon),2,horizon)])
                self.assertEqual(backend.accounting_version,4)
            with (directory/'mac_trace.csv').open() as stream:
                trace=list(csv.DictReader(stream))

        offered={r['uid']:r for r in trace
                 if r['event']=='OFFERED' and int(r['ac']) in (0,1)}
        outcomes={r['uid']:r for r in trace
                  if r['event'] in ('ACKED','DROP') and r['uid'] in offered}
        for ac,name,deadline_ms in ((0,'vo',20.),(1,'vi',50.)):
            eligible=on_time=0
            for uid,row in offered.items():
                if int(row['ac'])!=ac: continue
                created=float(row['time_s']); deadline=created+deadline_ms/1000
                outcome=outcomes.get(uid)
                outcome_time=float(outcome['time_s']) if outcome else float('inf')
                verdict_time=min(deadline,outcome_time)
                if verdict_time<=horizon+1e-12:
                    eligible+=1
                    on_time+=int(bool(outcome) and outcome['event']=='ACKED'
                                 and outcome_time<=deadline+1e-12)
            self.assertEqual(total[f'{name}_deadline_eligible_packets'],eligible)
            self.assertEqual(total[f'{name}_on_time_delivered_packets'],on_time)


if __name__=='__main__': unittest.main()
