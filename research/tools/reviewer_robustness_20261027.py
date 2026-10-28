"""Fixed-model robustness and expanding-history controls on already seen dates."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import sys,time,json,hashlib,copy
from pathlib import Path
sys.path[:0]=['/workspace','/workspace/condition_anchor_code_20261019/tools','/workspace/condition_anchor_code_20261019/src']
import numpy as np,pandas as pd,torch,joblib
from sklearn.ensemble import RandomForestRegressor
from reviewer_controls_20261027 import Control,write,nonoverlap,infer,STEPS,SEEDS
from run_plan_response_20261019 import data,encode,backbone
from plan_response_20261019 import PlanResponsePolicy,VARIANTS
from relational_ar_20261016 import masked_loss
from run_unified_baselines_20261019 import tabular_features
from run_conservative_budget_20261019 import measures
from replay_grpo_base_20261015 import sha
R=Path('/workspace');O=R/'reviewer_robustness_20261027'

def main():
    O.mkdir(exist_ok=False);torch.set_num_threads(4);start=time.monotonic();d=data(R);x=tabular_features(d);rows=[]
    write(O/'protocol.json',dict(seeds=SEEDS,runner_sha=sha(Path(__file__)),test_selection=False,independent_holdout=False,plan_changes='input corruption against unchanged observed trajectory, not counterfactual accuracy',folds=['2026-09-11 to 16','2026-09-16 to 21','2026-09-21 to 2026-10-01'],rolling_pool='original 4096 training issues plus strictly earlier evaluation issues with all target endpoints released before fold start',scalers='original pre-September11 only; unchanged',updates=STEPS,training_samples=4096,batch=64,scope='expanding-history supervised controls; not archived multistage training',resource_gate_seconds=21600))
    # Archived final weights and RF: identical corrupted plan slots and GT.
    for seed in SEEDS:
        base,_=backbone(R,seed,d['fm'].shape[1]);m=PlanResponsePolicy(base,VARIANTS[0]);m.load_state_dict(torch.load(R/'submission_raw_20261026/selected_training'/f'seed{seed}/final.pt',weights_only=False)['state']);m.eval();e=encode(m.backbone,d)
        rf=joblib.load(R/'unified_classical_20261019'/f'RandomForest__seed{seed}.joblib')
        for split in ['validation','test']:
            ids=d['splits'][split];pos=d['index'][ids]
            coverage=d['history'][pos,:,:,1].mean((1,2)).numpy();fuelcoverage=d['history'][pos,:,7,1].mean(1).numpy()
            for scenario in ['nominal','no_plan','speed_plus_3kn','plan_lag_1h']:
                p=d['plans'][0][pos].clone()
                if scenario=='no_plan':p[:,:,:5]=0
                if scenario=='speed_plus_3kn':p[:,:,0]=torch.where(p[:,:,3]>0,(p[:,:,0]+.06).clamp(0,1),p[:,:,0])
                if scenario=='plan_lag_1h':p[:,:,:5]=torch.cat([p[:,:1,:5].expand(-1,2,-1),p[:,:-2,:5]],1)
                with torch.inference_mode():
                    pred=np.concatenate([m.predict_encoded(tuple(v[pos[l:l+64]] for v in e),p[l:l+64]).numpy()*.3 for l in range(0,len(ids),64)])
                xx=x[pos].copy();xx[:,:,-9:]=p.numpy();rpred=np.maximum(rf.predict(xx.reshape(-1,xx.shape[-1])).reshape(-1,12),0)
                for name,pp in [('VORAFuel',pred),('RF',rpred)]:
                    if scenario=='nominal':
                        path=R/'submission_raw_20261026/selected_training'/f'seed{seed}/{split}.npz' if name=='VORAFuel' else R/'unified_classical_20261019'/f'{split}__RandomForest__seed{seed}.npz'
                        np.testing.assert_allclose(pp,np.load(path)['prediction'],atol=1e-6,rtol=0)
                    np.savez_compressed(O/f'{name}_{seed}_{split}_{scenario}.npz',prediction=pp,target=d['target'][ids],issues=ids)
                    for scope,at in [('all',np.arange(len(ids))),('nonoverlap',nonoverlap(ids)),('coverage_lt25',np.flatnonzero(coverage<.25)),('coverage_25_50',np.flatnonzero((coverage>=.25)&(coverage<.5))),('coverage_ge50',np.flatnonzero(coverage>=.5)),('fuel_missing_all',np.flatnonzero(fuelcoverage==0)),('fuel_some_observed',np.flatnonzero(fuelcoverage>0))]:
                        rows.extend(dict(name=name,seed=seed,split=split,scenario=scenario,scope=scope,**r) for r in measures(d,ids[at],pp[at]))
        pd.DataFrame(rows).to_csv(O/'fixed_robustness.csv',index=False)
    print('FIXED ROBUSTNESS COMPLETE',flush=True)
    folds=[('2026-09-11','2026-09-16'),('2026-09-16','2026-09-21'),('2026-09-21','2026-10-01')];rows=[];cost=[]
    for fold,(begin,end) in enumerate(folds):
        boundary=np.searchsorted(d['f'].index,pd.Timestamp(begin));endidx=np.searchsorted(d['f'].index,pd.Timestamp(end))
        pool=d['scope'][(d['release'][d['scope']].max(1)<boundary)&np.isfinite(d['target'][d['scope']]).any(1)]
        train=pool[np.linspace(0,len(pool)-1,4096,dtype=int)];assert len(np.unique(train))==4096
        ids=d['scope'][(d['scope']>=boundary)&(d['scope']+39<endidx)];pos=d['index'][ids];tp=d['index'][train]
        write(O/f'fold{fold}.json',dict(begin=begin,end=end,train_indices=train.tolist(),eval_indices=ids.tolist(),purge_release_verified=True))
        for seed in SEEDS:
            rng=np.random.default_rng(seed+27100);batches=[rng.choice(tp,64,replace=False) for _ in range(STEPS)]
            for kind in ['full','no_fm','gru_memory','RF']:
                assert time.monotonic()-start<21600
                tick=time.monotonic();dest=O/f'fold{fold}_{kind}_{seed}';dest.mkdir()
                if kind=='RF':
                    valid=np.isfinite(d['target'][train]);m=RandomForestRegressor(n_estimators=64,max_depth=12,min_samples_leaf=5,n_jobs=4,random_state=seed);m.fit(x[tp][valid],d['target'][train][valid]);pp=np.maximum(m.predict(x[pos].reshape(-1,x.shape[-1])).reshape(-1,12),0);joblib.dump(m,dest/'model.joblib')
                else:
                    torch.manual_seed(seed);m=Control(d['fm'].shape[1],kind);opt=torch.optim.Adam(m.parameters(),lr=.0003)
                    for at in batches:
                        loss=masked_loss(m(d['history'][at],d['fm'][at],d['plans'][0][at]),d['truth'][at]);assert torch.isfinite(loss)
                        opt.zero_grad();loss.backward();assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters());torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
                    m.eval();pp=infer(m,d,pos);torch.save(m.state_dict(),dest/'model.pt')
                assert np.isfinite(pp).all();np.savez_compressed(dest/'prediction.npz',prediction=pp,target=d['target'][ids],issues=ids)
                for scope,at in [('all',np.arange(len(ids))),('nonoverlap',nonoverlap(ids))]:rows.extend(dict(fold=fold,name=kind,seed=seed,scope=scope,**r) for r in measures(d,ids[at],pp[at]))
                cost.append(dict(fold=fold,name=kind,seed=seed,seconds=time.monotonic()-tick));pd.DataFrame(rows).to_csv(O/'rolling_metrics.csv',index=False);pd.DataFrame(cost).to_csv(O/'rolling_cost.csv',index=False)
                print(json.dumps(cost[-1]),flush=True)
    write(O/'completion.json',dict(complete=True,seconds=time.monotonic()-start,runs=len(cost),independent_holdout=False))

if __name__=='__main__':main()
