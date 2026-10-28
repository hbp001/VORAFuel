"""Preregistered scratch controls; no checkpoint selection or test tuning.

Same historical input, target release, training indices, batches and optimizer.
Availability controls exclude FM entirely, including its presence/age metadata.
These supervised controls do NOT replace the archived multistage VORAFuel.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import argparse, copy, fcntl, hashlib, json, sys, time
from pathlib import Path
sys.path[:0]=['/workspace/condition_anchor_code_20261019/tools','/workspace/condition_anchor_code_20261019/src']
import numpy as np
import pandas as pd
import torch
from torch import nn
from run_plan_response_20261019 import data
from operational_core import Policy
from relational_ar_20261016 import SixHourPolicy, ARCHS, masked_loss
from run_conservative_budget_20261019 import measures
from replay_grpo_base_20261015 import sha, state_sha

ARMS=['full','no_fm','no_relation','gru_memory','transformer_memory','values_only_no_fm','value_mask_no_fm']
SEEDS=[13,37,71]
STEPS=640
ROOT=Path('/workspace')

def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))

class SequentialMemory(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind;self.project=nn.Linear(42,24)
        self.position=nn.Embedding(36,24)
        self.layer=nn.GRU(24,24,batch_first=True) if kind=='gru_memory' else nn.TransformerEncoderLayer(24,4,48,dropout=0,batch_first=True,norm_first=True)
        self.norm=nn.LayerNorm(24)
    def forward(self,h):
        z=self.project(h.flip(1).flatten(2))+self.position.weight[None]
        z=self.layer(z)
        return self.norm(z[0] if self.kind=='gru_memory' else z)

class Control(SixHourPolicy):
    def __init__(self,nf,kind):
        super().__init__(Policy(288,216,nf),ARCHS[2]);self.kind=kind
        # Fresh full-length projections, not the legacy expanded initialization.
        self.vessel[0].reset_parameters();self.weather[0].reset_parameters()
        if kind in ['gru_memory','transformer_memory']: self.relation=SequentialMemory(kind)
    def encode(self,h,f):
        if self.kind=='values_only_no_fm': h=torch.cat([h[...,:1],torch.zeros_like(h[...,1:])],-1)
        if self.kind=='value_mask_no_fm': h=torch.cat([h[...,:2],torch.zeros_like(h[...,2:])],-1)
        c,r=super().encode(h,f)
        if self.kind.endswith('no_fm') or self.kind=='no_fm': c=torch.cat([c[:,:48],torch.zeros_like(c[:,48:])],1)
        if self.kind=='no_relation': r=torch.zeros_like(r)
        return c,r

def nonoverlap(ids):
    keep=[];last=-100000
    for j,i in enumerate(ids):
        if i>=last+36: keep.append(j);last=i
    return np.array(keep,dtype=int)

def infer(m,d,pos,scenario='nominal'):
    pieces=[]
    with torch.inference_mode():
        for left in range(0,len(pos),64):
            at=pos[left:left+64];p=d['plans'][0][at].clone()
            if scenario=='no_plan':p[:,:,:5]=0
            if scenario=='speed_plus_3kn':p[:,:,0]=torch.where(p[:,:,3]>0,(p[:,:,0]+.06).clamp(0,1),p[:,:,0])
            if scenario=='plan_lag_1h':p[:,:,:5]=torch.cat([p[:,:1,:5].expand(-1,2,-1),p[:,:-2,:5]],1)
            pieces.append((m(d['history'][at],d['fm'][at],p)*.3).numpy())
    return np.concatenate(pieces)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--pilot',action='store_true');a=parser.parse_args()
    a.out.mkdir(exist_ok=False);lock=(a.out/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.monotonic();d=data(ROOT)
    tp=d['index'][d['train']];assert (d['release'][d['train']]<=d['splits']['validation'][0]).all()
    protocol=dict(arms=ARMS,seeds=SEEDS,steps=STEPS,batch=64,lr=.0003,loss='masked Huber beta .2 + .15 MSE; no GRPO',init='all task-specific weights random; frozen past-only FM cache',availability_contrast=['values_only_no_fm','value_mask_no_fm','no_fm'],no_pretrained_task_checkpoint=True,train_indices_sha=hashlib.sha256(d['train'].tobytes()).hexdigest(),runner_sha=sha(Path(__file__)),selection=False,independent_holdout=False,scope='matched supervised scratch controls, not replacement of final multistage model',resource_gate_seconds=21600,pilot=a.pilot)
    write(a.out/'protocol.json',protocol);metrics=[];cost=[];pilot=[]
    for seed in ([13] if a.pilot else SEEDS):
        rng=np.random.default_rng(seed+27100);batches=np.stack([rng.choice(tp,64,replace=False) for _ in range(STEPS)])
        for kind in ARMS:
            assert time.monotonic()-start<21600,'suite 6h gate'
            torch.manual_seed(seed);m=Control(d['fm'].shape[1],kind);opt=torch.optim.Adam(m.parameters(),lr=.0003);initial=state_sha(m.state_dict());tick=time.monotonic();times=[]
            for k,at in enumerate(batches[:4] if a.pilot else batches):
                t=time.monotonic();loss=masked_loss(m(d['history'][at],d['fm'][at],d['plans'][0][at]),d['truth'][at]);assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
                nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();times.append(time.monotonic()-t)
                if k%64==63: print(json.dumps(dict(seed=seed,arm=kind,step=k+1,seconds=time.monotonic()-tick)),flush=True)
                assert time.monotonic()-start<21600,'suite 6h gate'
            assert all(torch.isfinite(p).all() for p in m.parameters());m.eval()
            if a.pilot:
                h=d['history'][tp[:2]].clone();f=d['fm'][tp[:2]].clone();p=d['plans'][0][tp[:2]]
                with torch.no_grad():
                    expected=m(h,f,p)
                    if kind.endswith('no_fm') or kind=='no_fm':torch.testing.assert_close(expected,m(h,f+100,p),atol=0,rtol=0)
                    if kind=='values_only_no_fm':h[...,1:]=100;torch.testing.assert_close(expected,m(h,f,p),atol=0,rtol=0)
                    if kind=='value_mask_no_fm':h[...,2]=100;torch.testing.assert_close(expected,m(h,f,p),atol=0,rtol=0)
                pilot.append(dict(arm=kind,step_seconds=float(np.median(times[1:])),estimated_train_seconds=float(np.median(times[1:])*STEPS)))
                continue
            dest=a.out/f'{kind}_seed{seed}';dest.mkdir();torch.save(dict(state=m.state_dict(),protocol=protocol),dest/'model.pt')
            cost.append(dict(arm=kind,seed=seed,steps=len(times),seconds=sum(times),parameters=sum(p.numel() for p in m.parameters()),initial_sha=initial,batch_sha=hashlib.sha256(batches.tobytes()).hexdigest()))
            for split in ['validation','test']:
                ids=d['splits'][split];pos=d['index'][ids]
                for scenario in ['nominal','no_plan','speed_plus_3kn','plan_lag_1h']:
                    pred=infer(m,d,pos,scenario);assert np.isfinite(pred).all()
                    np.savez_compressed(dest/f'{split}_{scenario}.npz',issues=ids,target=d['target'][ids],prediction=pred)
                    for scope,at in [('all',np.arange(len(ids))),('nonoverlap',nonoverlap(ids))]:
                        metrics.extend(dict(arm=kind,seed=seed,split=split,scenario=scenario,scope=scope,**r) for r in measures(d,ids[at],pred[at]))
            pd.DataFrame(metrics).to_csv(a.out/'metrics.csv',index=False);pd.DataFrame(cost).to_csv(a.out/'cost.csv',index=False)
            print('DONE '+kind+' '+str(seed),flush=True)
    if a.pilot:
        estimate=sum(x['estimated_train_seconds'] for x in pilot)*3
        write(a.out/'pilot.json',dict(arms=pilot,estimated_suite_training_seconds=estimate,gate_pass=estimate*1.3<21600,overhead_factor=1.3,invariance_tests_passed=True))
        assert estimate*1.3<21600
    else:write(a.out/'completion.json',dict(complete=True,runs=len(cost),seconds=time.monotonic()-start,adopted=False))

if __name__=='__main__': main()
