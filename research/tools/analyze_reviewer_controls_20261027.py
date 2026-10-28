"""Paired loss contrasts, provenance and missing-fuel strata; never select models."""
import sys,json,hashlib
from pathlib import Path
sys.path[:0]=['/workspace','/workspace/condition_anchor_code_20261019/tools','/workspace/condition_anchor_code_20261019/src']
import numpy as np,pandas as pd,torch
from reviewer_controls_20261027 import write,nonoverlap,SEEDS,ARMS
from run_plan_response_20261019 import data
from run_conservative_budget_20261019 import cohorts,measures
from replay_grpo_base_20261015 import sha,BASE,SOURCE_SHA,CACHE_SHA,SCALER_SHA
R=Path('/workspace');C=R/'reviewer_controls_full_20261027';B=R/'reviewer_rolling_v2_20261027';D=R/'reviewer_structure_detail_20261027';O=R/'reviewer_analysis_20261027'

def main():
    assert json.loads((C/'completion.json').read_text())['runs']==21
    assert json.loads((B/'completion.json').read_text())['runs']==24
    assert json.loads((D/'completion.json').read_text())['runs']==9
    O.mkdir(exist_ok=False);torch.set_num_threads(4);d=data(R);contrasts=[];missing=[];manifest=[]
    cost=pd.read_csv(C/'cost.csv');assert cost.groupby('seed').batch_sha.nunique().eq(1).all();assert cost.steps.eq(640).all()
    costs=pd.concat([cost,pd.read_csv(D/'cost.csv')]);assert costs.groupby('seed').batch_sha.nunique().eq(1).all()
    pd.concat([pd.read_csv(C/'metrics.csv'),pd.read_csv(D/'metrics.csv')]).to_csv(O/'all_controls_metrics.csv',index=False)
    def load(path,ids):
        z=np.load(path);np.testing.assert_array_equal(z['issues'],ids);np.testing.assert_allclose(z['target'],d['target'][ids],atol=0,rtol=0,equal_nan=True);assert np.isfinite(z['prediction']).all();manifest.append(dict(path=str(path),sha256=sha(path)));return z['prediction']
    for split in ['validation','test']:
        ids=d['splits'][split];gt=d['target'][ids];pred={a:np.stack([load(C/f'{a}_seed{s}/{split}_nominal.npz',ids) for s in SEEDS]) for a in ARMS}
        pred.update({a:np.stack([load(D/f'{a}_seed{s}/{split}_nominal.npz',ids) for s in SEEDS]) for a in ['no_time_block','no_variable_block','direct_decoder']})
        _,stay,depart,_=cohorts(d,ids)
        for arm,other in [('no_fm','values_only_no_fm'),('no_fm','value_mask_no_fm'),('value_mask_no_fm','values_only_no_fm'),('full','no_fm'),('full','no_relation'),('full','gru_memory'),('full','transformer_memory'),('full','no_time_block'),('full','no_variable_block'),('full','direct_decoder')]:
            for scope,at in [('all',np.arange(len(ids))),('nonoverlap',nonoverlap(ids))]:
                for metric in ['point','first','total','stay','departure']:
                    mask=np.isfinite(gt).copy()
                    if metric=='first':mask[:,1:]=False
                    if metric=='stay':mask &=stay
                    if metric=='departure':mask &=depart
                    n=mask[at].sum(1) if metric!='total' else np.isfinite(gt[at]).all(1).astype(int)
                    def loss(pp):
                        if metric=='total':return np.where(n>0,np.abs(np.nansum(pp[:,at]-gt[at],2)).mean(0),0)
                        return np.where(mask[at],np.abs(pp[:,at]-gt[at]).mean(0),0).sum(1)
                    delta=loss(pred[arm])-loss(pred[other])
                    for block in [1,2]:
                        labels=(d['f'].index[ids[at]].normalize()-pd.Timestamp('2026-09-11')).days//block
                        df=pd.DataFrame(dict(block=labels,delta=delta,n=n)).groupby('block').sum();df=df[df.n>0]
                        if not len(df):continue
                        rng=np.random.default_rng(271027);ix=rng.integers(len(df),size=(10000,len(df)));v=df.delta.to_numpy()[ix].sum(1)/df.n.to_numpy()[ix].sum(1)
                        contrasts.append(dict(split=split,arm=arm,comparator=other,scope=scope,metric=metric,block_days=block,clusters=len(df),n=int(df.n.sum()),delta=float(df.delta.sum()/df.n.sum()),lo=float(np.quantile(v,.025)),hi=float(np.quantile(v,.975))))
        # Same naturally missing input strata for final model, RF and each FM.
        coverage=d['history'][d['index'][ids],:,7,1].mean(1).numpy()
        paths={
            'VORAFuel':[R/'submission_raw_20261026/selected_training'/f'seed{s}/{split}.npz' for s in SEEDS],
            'RF':[R/'unified_classical_20261019'/f'{split}__RandomForest__seed{s}.npz' for s in SEEDS],
            'Persistence':[R/'unified_classical_20261019'/f'{split}__최근관측지속__seed0.npz'],
            **{k:[R/'unified_foundation_20261019'/f'{split}__{v}__seed0.npz'] for k,v in [('Chronos2','Chronos2단독_연료'),('Chronos2_cov','Chronos2단독_공변량계획'),('TimesFM','TimesFM2.5단독_연료')]}}
        for name,files in paths.items():
            for j,path in enumerate(files):
                pp=load(path,ids)
                for group,at in [('fuel_missing_all',np.flatnonzero(coverage==0)),('fuel_some_observed',np.flatnonzero(coverage>0))]:missing.extend(dict(name=name,seed=SEEDS[j] if len(files)>1 else 0,split=split,group=group,**r) for r in measures(d,ids[at],pp[at]))
    pd.DataFrame(contrasts).to_csv(O/'paired_bootstrap.csv',index=False);pd.DataFrame(missing).to_csv(O/'natural_missing.csv',index=False)
    rolling=pd.read_csv(B/'metrics.csv');extra=[]
    for fold in [0,1]:
        info=json.loads((B/f'fold{fold}.json').read_text());ids=np.array(info['eval_indices']);split='validation' if fold==0 else 'test'
        for name,path in [
            ('Persistence',R/'unified_classical_20261019'/f'{split}__최근관측지속__seed0.npz'),
            *[(k,R/'unified_foundation_20261019'/f'{split}__{v}__seed0.npz') for k,v in [('Chronos2','Chronos2단독_연료'),('Chronos2_cov','Chronos2단독_공변량계획'),('TimesFM','TimesFM2.5단독_연료')]]]:
            original=d['splits'][split];pp=load(path,original);at=np.searchsorted(original,ids);np.testing.assert_array_equal(original[at],ids);pp=pp[at]
            for scope,ix in [('all',np.arange(len(ids))),('nonoverlap',nonoverlap(ids))]:extra.extend(dict(fold=fold,name=name,seed=0,scope=scope,**r) for r in measures(d,ids[ix],pp[ix]))
    pd.concat([rolling,pd.DataFrame(extra)]).to_csv(O/'rolling_with_fm.csv',index=False)
    robustness=list((R/'reviewer_robustness_20261027').glob('*.npz'));assert len(robustness)==48
    for path in robustness:
        split='validation' if '_validation_' in path.name else 'test';load(path,d['splits'][split])
    for directory in [C,B,D,R/'reviewer_robustness_20261027']:
        manifest.extend(dict(path=str(p),sha256=sha(p)) for p in directory.rglob('*') if p.is_file() and p.suffix in ['.json','.csv','.py','.pt','.joblib','.npz'])
    lineage={'source':R/'operational_qc_v3_regenerated_20261013/merged_qc_frame.csv.gz','cache':R/'fresh_fm_cache_20261013/fresh_dual_cache.npz','scaler':R/BASE/'scalers.pt'}
    hashes={k:sha(v) for k,v in lineage.items()};assert list(hashes.values())==[SOURCE_SHA,CACHE_SHA,SCALER_SHA]
    write(O/'source_audit.json',dict(hashes=hashes,code={p.name:sha(p) for p in (R/'condition_anchor_code_20261019/src').glob('*.py')},target_release_checked_before_training=True))
    write(O/'manifest.json',manifest);write(O/'completion.json',dict(complete=True,runs=54,all_seed_batches_identical_within_comparison=True,source_target_identity=True,finite_predictions=True,bootstrap=10000,block_days=[1,2],multiplicity='marginal descriptive intervals; no omnibus significance claim',independent_holdout=False,model_adoption=False))

if __name__=='__main__':main()
