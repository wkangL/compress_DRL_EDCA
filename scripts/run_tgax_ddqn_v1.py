"""First small, single-seed TGax TCN-DDQN closed-loop validation."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import csv,json,math,random,time
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT,load_config
from models.branching_ddqn import ControlNetwork
from reporting.sta_state import StaHistory,StateScaler
from rl.replay_buffer import ReplayBuffer,Transition
from rl.trainer import InferenceCache,explore,freeze_target,set_seed,td_update,update_target
from rl.tgax_environment import TgaxControlEnvironment


def baseline_scaler(config:dict,path:Path)->StateScaler:
    n=config['network']['num_sta']; window=round(config['timing']['history_s']/config['timing']['sta_sample_s'])
    history=StaHistory(n,window); samples=[]; last=None
    for line in path.read_text(encoding='utf-8').splitlines():
        f=line.split()
        if not f or f[0]!='S' or int(float(f[2]))<=0: continue
        t=float(f[1]); node=int(float(f[2])); ac=int(float(f[3]))
        history.append(node-1,ac,t,np.asarray(list(map(float,f[4:10])),dtype=np.float32))
        if ac==3 and node==n and t!=last: samples.append(history.all()); last=t
    if not samples: raise RuntimeError('No baseline state samples for scaler')
    scaler=StateScaler(config['state']['delay_clip_ms']); scaler.fit(np.stack(samples)); return scaler


def save_csv(path:Path,rows:list[dict]):
    if not rows:return
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def compact(metrics):
    return {k:(json.dumps(v) if isinstance(v,(list,dict)) else v) for k,v in metrics.items()
            if not k.endswith('_delay_samples_ms')}


def aggregate(rows:list[dict])->dict:
    result={}
    for ac in ('vo','vi','be','bk'):
        for direction in ('ul','dl'):
            prefix=f'{direction}_{ac}'
            samples=sum((r[f'{prefix}_delay_samples_ms'] for r in rows),[])
            arrived=sum(r[f'{prefix}_arrived'] for r in rows); failed=sum(r[f'{prefix}_failed'] for r in rows)
            attempts=sum(r[f'{prefix}_tx_attempts'] for r in rows); retries=sum(r[f'{prefix}_retransmissions'] for r in rows)
            result.update({f'{prefix}_delay_p95_ms':float(np.percentile(samples,95)) if samples else None,
                f'{prefix}_throughput_mbps':float(np.mean([r[f'{prefix}_goodput_mbps'] for r in rows])),
                f'{prefix}_final_drop_rate':failed/arrived if arrived else None,
                f'{prefix}_retransmission_rate':retries/attempts if attempts else None,
                f'{prefix}_successful_samples':len(samples)})
    result['reward_mean']=float(np.mean([r['reward'] for r in rows])); return result


def jain_from_flows(run:Path,flows_path:Path)->float|None:
    if not (run/'flow_summary.csv').is_file(): return None
    assignments={int(r['flow_id']):int(r['sta']) for r in csv.DictReader(flows_path.open(encoding='utf-8')) if int(r['bss'])==0}
    offered=Counter(); received=Counter()
    for r in csv.DictReader((run/'flow_summary.csv').open(encoding='utf-8')):
        if int(r['bss'])!=0 or int(r['flow_id']) not in assignments: continue
        sta=assignments[int(r['flow_id'])]; offered[sta]+=int(r['offered_bytes']); received[sta]+=int(r['received_bytes'])
    values=np.asarray([min(received[s]/offered[s],1.) for s in offered if offered[s]>0])
    return float(values.sum()**2/(len(values)*np.square(values).sum())) if len(values) and np.square(values).sum()>0 else None


def evaluate(config,model,scaler,seed,path,learned):
    duration=config['evaluation']['duration_s']; env=TgaxControlEnvironment(config,'compressed_async_age' if learned else 'default',seed,path,duration)
    rows=[]; inference=InferenceCache(model,scaler) if learned else None; obs=env.reset_observation()
    try:
        for step in range(round(duration/config['timing']['decision_s'])):
            action=inference.select(obs) if learned else np.array([0,0,5,5,2])
            obs,reward,metrics=env.step(action,config['timing']['decision_s'],custom=learned)
            rows.append({'step':step+1,'action':action.tolist(),'reward':reward,**metrics})
    finally: env.close()
    summary=aggregate(rows); summary['jain_demand_satisfaction']=jain_from_flows(path,ROOT/config['backend']['flows'])
    (path/'evaluation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    save_csv(path/'evaluation_steps.csv',[{'step':r['step'],'action':json.dumps(r['action']),**compact(r)} for r in rows])
    return summary,rows


def main():
    config=load_config(ROOT/'configs/tgax_ddqn_v1.yaml'); seed=config['run']['seeds'][0]
    root=ROOT/'results/raw/tgax_ddqn_v1_small_seed41_r4_20261023'
    if root.exists(): raise SystemExit(f'Refusing to overwrite {root}')
    root.mkdir(parents=True); (root/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    set_seed(seed,config['training']['threads']); rng=np.random.default_rng(seed)
    scaler=baseline_scaler(config,ROOT/'results/raw/tgax_ddqn_v1_fixed_assoc_default_step_5s_20261023/events.log')
    model=ControlNetwork(config,'compressed_async_age',10); target=freeze_target(deepcopy(model))
    tc=config['training']; optimizer=torch.optim.AdamW([{'params':model.q.parameters(),'lr':tc['q_lr']},
        {'params':list(model.encoder.parameters())+list(model.aggregator.parameters()),'lr':tc['representation_lr']}])
    replay=ReplayBuffer(tc['replay_capacity'],seed); steps=updates=version=0; step_rows=[]; losses=[]; action_counts=[Counter() for _ in range(5)]
    started=time.perf_counter(); status={'state':'training','seed':seed,'started_at':datetime.now().isoformat()}
    (root/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    try:
        for stage in range(tc['stages']):
            env=TgaxControlEnvironment(config,'compressed_async_age',seed*100+stage+1,root/f'train_stage_{stage+1}',tc['episode_s'],version)
            obs=env.reset_observation(); inference=InferenceCache(model,scaler)
            try:
                for local in range(round(tc['episode_s']/config['timing']['decision_s'])):
                    epsilon=tc['epsilon_end']+(tc['epsilon_start']-tc['epsilon_end'])*max(0,1-steps/tc['epsilon_steps'])
                    action=explore(model,rng) if rng.random()<epsilon else inference.select(obs)
                    following,reward,metrics=env.step(action,config['timing']['decision_s'],custom=True)
                    replay.add(Transition(obs,action,reward,following,False,version,local+1==tc['episode_s'],
                        'candidate_voip8_gaming4_ftp_bk',True)); obs=following; steps+=1
                    for i,a in enumerate(action): action_counts[i][int(a)]+=1
                    step_rows.append(dict(stage=stage+1,step=steps,epsilon=epsilon,action=json.dumps(action.tolist()),**compact(metrics)))
                    print(json.dumps({'stage':stage+1,'step':steps,'epsilon':epsilon,'reward':reward}),flush=True)
            finally: env.close()
            if len(replay)>=max(tc['warmup'],tc['batch_size']):
                model.train()
                for _ in range(tc['updates_per_stage']):
                    stats=td_update(model,target,replay.sample(tc['batch_size']),scaler,optimizer,tc); updates+=1
                    synced,before,after=update_target(model,target,updates,tc)
                    stats.update(stage=stage+1,update=updates,target_sync=synced,target_distance_before=before,target_distance_after=after)
                    losses.append(stats)
                    if max(abs(stats['q_min']),abs(stats['q_max']),abs(stats['target_q_min']),abs(stats['target_q_max']))>1e4 or stats['grad_norm']>1e3:
                        raise FloatingPointError('TGax training divergence threshold exceeded')
                version+=1; model.eval()
            save_csv(root/'training_steps.csv',step_rows); save_csv(root/'losses.csv',losses)
            torch.save({'model':model.state_dict(),'target':target.state_dict(),'optimizer':optimizer.state_dict(),
                'scaler_mean':scaler.mean,'scaler_std':scaler.std,'config':config,'seed':seed,'steps':steps,
                'updates':updates,'encoder_version':version},root/'checkpoint.pt')
        default_summary,_=evaluate(config,model,scaler,seed+1000,root/'eval_default',False)
        learned_summary,learned_rows=evaluate(config,model,scaler,seed+1000,root/'eval_learned',True)
        distribution=[dict(branch=i,counts=dict(sorted(c.items()))) for i,c in enumerate(action_counts)]
        result={'state':'complete','steps':steps,'updates':updates,'runtime_s':time.perf_counter()-started,
            'default':default_summary,'learned':learned_summary,'training_action_distribution':distribution,
            'evaluation_actions':[r['action'] for r in learned_rows]}
        (root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        (root/'status.json').write_text(json.dumps({'state':'complete','seed':seed,'steps':steps,
            'updates':updates,'finished_at':datetime.now().isoformat()},indent=2),encoding='utf-8')
    except BaseException as error:
        diagnostic={'state':'aborted','error':repr(error),'steps':steps,'updates':updates,
            'last_losses':losses[-20:],'runtime_s':time.perf_counter()-started}
        (root/'diagnostic_abort.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
        (root/'status.json').write_text(json.dumps({'state':'aborted','seed':seed,'steps':steps,
            'updates':updates,'error':repr(error),'finished_at':datetime.now().isoformat()},indent=2),encoding='utf-8')
        raise

if __name__=='__main__': main()
