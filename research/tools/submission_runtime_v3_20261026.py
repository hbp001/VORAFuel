"""Matched CPU inference audit, fresh request preprocessing and cache refresh.

No fitting, selection, future fuel input, or deployment. Batch size one.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import sys, json, time, gc, fcntl
from pathlib import Path
sys.path[:0]=['/workspace/condition_anchor_code_20261019/tools','/workspace/condition_anchor_code_20261019/src']
import numpy as np
import pandas as pd
import torch, joblib
from run_plan_response_20261019 import data, backbone
from plan_response_20261019 import PlanResponsePolicy, VARIANTS
from replay_grpo_base_20261015 import BASE, sha
from operational_core import causal_channels, VESSEL, WEATHER
from planned_multiscale_20261015 import plan_tokens, MODES
from run_unified_baselines_20261019 import foundation_inputs, aggregate_native10, tabular_features

R=Path('/workspace'); O=R/'submission_runtime_v3_20261026'
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def main():
    O.mkdir(exist_ok=False);lock=(O/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);d=data(R)
    sc=torch.load(R/BASE/'scalers.pt',weights_only=False)
    with np.load(R/'fresh_fm_cache_20261013/fresh_dual_cache.npz') as archived:
        cache={k:archived[k] for k in ['x','group_present','issue_indices']}
    ids=d['splits']['validation'][np.linspace(0,len(d['splits']['validation'])-1,12,dtype=int)]
    save(O/'protocol.json',dict(issues=ids.tolist(),repetitions=3,warmup=2,threads=4,batch=1,source_sha=sha(R/'operational_qc_v3_regenerated_20261013/merged_qc_frame.csv.gz'),boundary='in-memory raw prefix -> causal channels, scaling, plan construction, model -> 12 bin prediction; disk/network/model loading excluded; prefix last200 rows',refresh='Ours cache-hit and forced regeneration of the same scheduled past cache separately',selection=False))
    rows=[];checks=[]
    def prepare(issue):
        # 200 rows exceed the 144-step age cap, plan lag, and forward-fill span.
        f=d['f'].iloc[issue-199:issue+1].copy();j=len(f)-1
        # Empty future rows expose only timestamps for plan clock features.
        future=pd.DataFrame(index=pd.date_range(f.index[-1]+pd.Timedelta(minutes=10),periods=144,freq='10min'),columns=f.columns)
        pp=torch.tensor(plan_tokens(pd.concat([f,future]),np.array([j]),MODES[1])[:,1:13])
        blocks=causal_channels(f,3);hs=[]
        for (v,m,a),s in zip(blocks,sc['channel_scalers']):
            z=np.nan_to_num((v-np.asarray(s['median']))/np.asarray(s['std'])).clip(-10,10)
            hs.append(np.stack([z,m,a],-1)[j-np.arange(36)])
        h=torch.tensor(np.concatenate(hs,1)[None],dtype=torch.float32)
        v,m,_=blocks[0];w,wm,_=blocks[1]
        raw=np.array([v[j,0]/50,v[j,7]*10,v[j,5],v[j,6],w[j,2]/10,w[j,3]/10,w[j,0]/3,w[j,4],w[j,5],m[j,0],m[j,7],m[j,5]*m[j,6],wm[j,2]*wm[j,3],wm[j,0]*wm[j,4]*wm[j,5]])
        return f,blocks,h,pp,torch.tensor(np.nan_to_num(raw)[None],dtype=torch.float32)
    def fmvalue(issue,blocks,refresh,pipe):
        k=np.searchsorted(cache['issue_indices'],issue,side='right')-1
        if refresh:
            local=199-(issue-int(cache['issue_indices'][k]));inp=[];present=[]
            for v,_,_ in blocks:
                b=v[local-11:local+1].astype(np.float32).T.copy();ok=np.isfinite(b).any();present.append(ok)
                if not ok:b[0,-1]=0
                inp.append(b)
            enc,_=pipe.embed(inp,batch_size=32,context_length=12)
            z=np.concatenate([e.float().mean((0,1)).numpy() if ok else np.zeros(768) for e,ok in zip(enc,present)])
        else:z=cache['x'][k];present=cache['group_present'][k]
        a=(issue-int(cache['issue_indices'][k]))/6
        return torch.tensor(np.r_[((z-sc['fm_mean'])/sc['fm_std']).clip(-10,10),present,a][None],dtype=torch.float32)
    def record(name,fn,expected):
        for issue in ids[:2]:fn(int(issue))
        for rep in range(3):
            for issue in ids:
                tick=time.perf_counter();y=fn(int(issue));sec=time.perf_counter()-tick
                assert y.shape==(12,) and np.isfinite(y).all()
                err=float(np.max(np.abs(y-expected[int(issue)])))
                rows.append(dict(model=name,rep=rep,issue=int(issue),seconds=sec,max_saved_difference=err))
        pd.DataFrame(rows).to_csv(O/'samples.csv',index=False)
        print('DONE',name,flush=True)
    with torch.inference_mode():
        from chronos import Chronos2Pipeline
        pipe=Chronos2Pipeline.from_pretrained('amazon/chronos-2',device_map='cpu',dtype=torch.float32,local_files_only=True)
        for seed in [13,37,71]:
            b,_=backbone(R,seed,d['fm'].shape[1]);model=PlanResponsePolicy(b,VARIANTS[0]);cp=R/'plan_response_full_v2_20261019'/f'seed{seed}'/'인과계획학습__동일예산기존형__GRPO고정.pt'
            model.load_state_dict(torch.load(cp,weights_only=False)['state']);model.eval()
            pred=np.load(R/'final_environment_full_20261021'/f'seed{seed}'/'validation__기존고정.npz');expected=dict(zip(pred['issues'],pred['prediction']))
            for refresh in [False,True]:
                def ours(issue):
                    _,bl,h,p,raw=prepare(issue);fm=fmvalue(issue,bl,refresh,pipe);c,m=model.backbone.encode(h,fm)
                    return model.predict_encoded((c,m,raw,torch.zeros(1,1)),p).numpy()[0]*.3
                record('Ours_'+('refresh' if refresh else 'hit')+f'_seed{seed}',ours,expected)
            rf=joblib.load(R/'unified_classical_20261019'/f'RandomForest__seed{seed}.joblib')
            pred=np.load(R/'unified_classical_20261019'/f'validation__RandomForest__seed{seed}.npz');expected=dict(zip(pred['issues'],pred['prediction']))
            def forest(issue):
                _,_,h,p,raw=prepare(issue);a=h.numpy();v=a[:,:,:,0];m=a[:,:,:,1];age=a[:,:,:,2]
                summary=np.c_[v[:,0],v.mean(1),v.std(1),v[:,0]-v[:,-1],m.mean(1),age[:,0],raw.numpy()]
                x=np.concatenate([np.repeat(summary[:,None],12,1),p.numpy()],2).astype(np.float32)
                return rf.predict(x[0])
            record(f'RF_seed{seed}',forest,expected)
            checks.append(dict(seed=seed,checkpoint_sha=sha(cp),parameters=sum(p.numel() for p in model.parameters())))
        median=float(np.nanmedian(d['received'][d['train'],7]))
        def native(issue,cov):
            f,_,_,p,_=prepare(issue)
            local=dict(f=f,plans=[p],index=np.r_[np.full(199,-1),0])
            uni,cc,_=foundation_inputs(local,np.array([199]),median)
            qq,_=pipe.predict_quantiles(cc if cov else uni,prediction_length=39,context_length=36,quantile_levels=[.5],batch_size=1,cross_learning=False)
            return np.maximum(aggregate_native10(qq[0][0,:,0].float().numpy()[None])[0],0)
        for label,cov in [('Chronos2단독_연료',False),('Chronos2단독_공변량계획',True)]:
            pred=np.load(R/'unified_foundation_20261019'/f'validation__{label}__seed0.npz');record(label,lambda i:native(i,cov),dict(zip(pred['issues'],pred['prediction'])))
        del pipe;gc.collect()
        import timesfm
        model=timesfm.TimesFM_2p5_200M_torch.from_pretrained('google/timesfm-2.5-200m-pytorch',local_files_only=True)
        model.compile(timesfm.ForecastConfig(max_context=64,max_horizon=64,normalize_inputs=True,use_continuous_quantile_head=True,force_flip_invariance=True,infer_is_positive=True,fix_quantile_crossing=True,return_backcast=False,per_core_batch_size=1))
        def tfm(issue):
            _,blocks,_,_,_=prepare(issue);v=blocks[0][0][-36:,7].astype(np.float32);v=np.where(np.isfinite(v),v,median)
            return np.maximum(aggregate_native10(model.forecast(horizon=39,inputs=[v])[0])[0],0)
        pred=np.load(R/'unified_foundation_20261019'/'validation__TimesFM2.5단독_연료__seed0.npz');record('TimesFM2.5',tfm,dict(zip(pred['issues'],pred['prediction'])))
    tab=pd.DataFrame(rows);summary=tab.groupby('model').seconds.agg(['count','median','mean','max']);summary['p95']=tab.groupby('model').seconds.quantile(.95);summary['max_prediction_difference']=tab.groupby('model').max_saved_difference.max();summary.to_csv(O/'summary.csv')
    save(O/'completion.json',dict(complete=True,checks=checks,parameters_exclude_frozen_FM=True));print(summary.to_string(),flush=True)
if __name__=='__main__':main()
