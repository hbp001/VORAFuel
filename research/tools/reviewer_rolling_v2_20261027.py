"""Two expanding-history folds; failed boundary-count audit retained in v1.

September 11 fixed-origin controls are evaluated by reviewer_controls separately.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import sys,time,hashlib
from pathlib import Path
sys.path[:0]=['/workspace','/workspace/condition_anchor_code_20261019/tools','/workspace/condition_anchor_code_20261019/src']
import numpy as np,pandas as pd,torch,joblib
from sklearn.ensemble import RandomForestRegressor
from reviewer_controls_20261027 import Control,write,nonoverlap,infer,STEPS,SEEDS
from run_plan_response_20261019 import data
from relational_ar_20261016 import masked_loss
from run_unified_baselines_20261019 import tabular_features
from run_conservative_budget_20261019 import measures
from replay_grpo_base_20261015 import sha
R=Path('/workspace');O=R/'reviewer_rolling_v2_20261027'

def main():
    O.mkdir(exist_ok=False);torch.set_num_threads(4);start=time.monotonic();d=data(R);x=tabular_features(d);rows=[];cost=[]
    write(O/'protocol.json',dict(seeds=SEEDS,runner_sha=sha(Path(__file__)),controls_sha=sha(R/'reviewer_controls_20261027.py'),selection=False,independent_holdout=False,folds=['2026-09-16 to 21','2026-09-21 to 2026-10-01'],rolling_pool='original training sample plus earlier evaluation windows; all target endpoints strictly released before boundary',scalers='original pre-September11 only',updates=STEPS,training_samples=4096,batch=64,lr=.0003,init='random task-specific weights; same frozen past FM',failure_preserved='reviewer_robustness_20261027: first fold sample count below4096 after strict release purge; fixed-model results valid, no rolling training started',scope='supervised expanding-history diagnostic on previously seen dates'))
    for fold,(begin,end) in enumerate([('2026-09-16','2026-09-21'),('2026-09-21','2026-10-01')]):
        boundary=np.searchsorted(d['f'].index,pd.Timestamp(begin));endidx=np.searchsorted(d['f'].index,pd.Timestamp(end))
        pool=d['scope'][(d['release'][d['scope']].max(1)<boundary)&np.isfinite(d['target'][d['scope']]).any(1)]
        assert len(pool)>=4096;train=pool[np.linspace(0,len(pool)-1,4096,dtype=int)];assert len(np.unique(train))==4096
        ids=d['scope'][(d['scope']>=boundary)&(d['scope']+39<endidx)];pos=d['index'][ids];tp=d['index'][train]
        write(O/f'fold{fold}.json',dict(begin=begin,end=end,train_indices=train.tolist(),eval_indices=ids.tolist(),purge_release_verified=True,pool_size=len(pool)))
        for seed in SEEDS:
            rng=np.random.default_rng(seed+27100);batches=[rng.choice(tp,64,replace=False) for _ in range(STEPS)]
            for kind in ['full','no_fm','gru_memory','RF']:
                assert time.monotonic()-start<21600;tick=time.monotonic();dest=O/f'fold{fold}_{kind}_{seed}';dest.mkdir()
                if kind=='RF':
                    valid=np.isfinite(d['target'][train]);m=RandomForestRegressor(n_estimators=64,max_depth=12,min_samples_leaf=5,n_jobs=4,random_state=seed);m.fit(x[tp][valid],d['target'][train][valid]);pp=np.maximum(m.predict(x[pos].reshape(-1,x.shape[-1])).reshape(-1,12),0);joblib.dump(m,dest/'model.joblib')
                else:
                    torch.manual_seed(seed);m=Control(d['fm'].shape[1],kind);opt=torch.optim.Adam(m.parameters(),lr=.0003)
                    for at in batches:
                        assert time.monotonic()-start<21600
                        loss=masked_loss(m(d['history'][at],d['fm'][at],d['plans'][0][at]),d['truth'][at]);assert torch.isfinite(loss)
                        opt.zero_grad();loss.backward();assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters());torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
                    m.eval();pp=infer(m,d,pos);torch.save(m.state_dict(),dest/'model.pt')
                assert np.isfinite(pp).all();np.savez_compressed(dest/'prediction.npz',prediction=pp,target=d['target'][ids],issues=ids)
                for scope,at in [('all',np.arange(len(ids))),('nonoverlap',nonoverlap(ids))]:rows.extend(dict(fold=fold,name=kind,seed=seed,scope=scope,**r) for r in measures(d,ids[at],pp[at]))
                cost.append(dict(fold=fold,name=kind,seed=seed,seconds=time.monotonic()-tick,batch_sha=hashlib.sha256(np.stack(batches).tobytes()).hexdigest()));pd.DataFrame(rows).to_csv(O/'metrics.csv',index=False);pd.DataFrame(cost).to_csv(O/'cost.csv',index=False)
                print(str(cost[-1]),flush=True)
    write(O/'completion.json',dict(complete=True,seconds=time.monotonic()-start,runs=len(cost),independent_holdout=False))

if __name__=='__main__':main()
