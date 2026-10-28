"""Separate time block, variable block and autoregression at the same SFT budget."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import sys,time,hashlib
from pathlib import Path
sys.path[:0]=['/workspace','/workspace/condition_anchor_code_20261019/tools','/workspace/condition_anchor_code_20261019/src']
import numpy as np,pandas as pd,torch
from reviewer_controls_20261027 import Control,write,nonoverlap,infer,STEPS,SEEDS
from run_plan_response_20261019 import data
from relational_ar_20261016 import ARCHS,masked_loss
from run_conservative_budget_20261019 import measures
from replay_grpo_base_20261015 import sha
R=Path('/workspace');O=R/'reviewer_structure_detail_20261027'

def main():
    O.mkdir(exist_ok=False);torch.set_num_threads(4);start=time.monotonic();d=data(R);tp=d['index'][d['train']];rows=[];cost=[]
    write(O/'protocol.json',dict(arms=['no_time_block','no_variable_block','direct_decoder'],seeds=SEEDS,steps=STEPS,batch=64,lr=.0003,runner_sha=sha(Path(__file__)),base_code_sha=sha(R/'reviewer_controls_20261027.py'),reference='reviewer_controls_full_20261027/full',selection=False,independent_holdout=False,whole_attention_blocks_removed=True,training_indices_sha=hashlib.sha256(d['train'].tobytes()).hexdigest()))
    for seed in SEEDS:
        rng=np.random.default_rng(seed+27100);batches=np.stack([rng.choice(tp,64,replace=False) for _ in range(STEPS)])
        for kind in ['no_time_block','no_variable_block','direct_decoder']:
            torch.manual_seed(seed);m=Control(d['fm'].shape[1],'full')
            if kind=='no_time_block':m.relation.time_layer=torch.nn.Identity()
            if kind=='no_variable_block':m.relation.variable_layer=torch.nn.Identity()
            if kind=='direct_decoder':m.architecture=ARCHS[1]
            opt=torch.optim.Adam(m.parameters(),lr=.0003);tick=time.monotonic()
            for k,at in enumerate(batches):
                assert time.monotonic()-start<21600
                loss=masked_loss(m(d['history'][at],d['fm'][at],d['plans'][0][at]),d['truth'][at]);assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters());torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
                if k==3:assert (time.monotonic()-tick)/4*STEPS<21600
            m.eval();dest=O/f'{kind}_seed{seed}';dest.mkdir();torch.save(dict(state=m.state_dict(),kind=kind),dest/'model.pt')
            for split in ['validation','test']:
                ids=d['splits'][split];pp=infer(m,d,d['index'][ids]);assert np.isfinite(pp).all();np.savez_compressed(dest/f'{split}_nominal.npz',issues=ids,target=d['target'][ids],prediction=pp)
                for scope,at in [('all',np.arange(len(ids))),('nonoverlap',nonoverlap(ids))]:rows.extend(dict(arm=kind,seed=seed,split=split,scenario='nominal',scope=scope,**r) for r in measures(d,ids[at],pp[at]))
            cost.append(dict(arm=kind,seed=seed,steps=STEPS,seconds=time.monotonic()-tick,batch_sha=hashlib.sha256(batches.tobytes()).hexdigest()));pd.DataFrame(rows).to_csv(O/'metrics.csv',index=False);pd.DataFrame(cost).to_csv(O/'cost.csv',index=False);print(str(cost[-1]),flush=True)
    write(O/'completion.json',dict(complete=True,runs=len(cost),seconds=time.monotonic()-start,adopted=False))

if __name__=='__main__':main()
