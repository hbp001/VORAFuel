"""Native foundation and classical baselines on ALL final six-hour issues."""
import argparse,gc,hashlib,json,os,sys,time,fcntl
from pathlib import Path
import numpy as np
import pandas as pd
import torch,joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_plan_response_20261019 import data
from run_conservative_budget_20261019 import measures,dump
from operational_core import causal_channels,VESSEL,WEATHER
from replay_grpo_base_20261015 import sha

def tabular_features(d):
    h=d['history'].numpy();v=h[:,:,:,0];mask=h[:,:,:,1];age=h[:,:,:,2]
    summary=np.c_[v[:,0],v.mean(1),v.std(1),v[:,0]-v[:,-1],mask.mean(1),age[:,0],d['raw'].numpy()]
    return np.concatenate([np.repeat(summary[:,None],12,1),d['plans'][0].numpy()],2).astype(np.float32)

def aggregate_native10(prediction):
    prediction=np.asarray(prediction);assert prediction.ndim==2 and prediction.shape[1]==39
    return prediction[:,3:].reshape(-1,12,3).sum(2)

def foundation_inputs(d,ids,median):
    blocks=causal_channels(d['f'],3);values=np.c_[blocks[0][0],blocks[1][0]];masks=np.c_[blocks[0][1],blocks[1][1]];ages=np.c_[blocks[0][2],blocks[1][2]];columns=list(VESSEL)+list(WEATHER);uni=[];cov=[];empty=0
    for issue in ids:
        past=issue-np.arange(35,-1,-1);v=values[past];target=v[:,7].astype(np.float32).copy()
        if not np.isfinite(target).any():target[-1]=median;empty+=1
        pc={columns[k]:v[:,k].astype(np.float32) for k in range(14) if k!=7}
        for k in range(14):pc[columns[k]+'_mask']=masks[past,k].astype(np.float32);pc[columns[k]+'_age']=ages[past,k].astype(np.float32)
        stamp=d['f'].index[past]-pd.Timedelta(minutes=30);pc['hour_sin']=np.sin(2*np.pi*(stamp.hour.to_numpy()+stamp.minute.to_numpy()/60)/24).astype(np.float32);pc['hour_cos']=np.cos(2*np.pi*(stamp.hour.to_numpy()+stamp.minute.to_numpy()/60)/24).astype(np.float32)
        plan=d['plans'][0][d['index'][issue]].numpy();future={}
        for k,pk in [(0,0),(5,1),(6,2)]:
            z=plan[:,pk]*(50 if k==0 else 1);valid=plan[:,3 if k==0 else 4]>0;z=np.where(valid,z,np.nan);future[columns[k]]=np.r_[np.repeat(v[-1,k],3),np.repeat(z,3)].astype(np.float32)
        times=pd.date_range(d['f'].index[issue]-pd.Timedelta(minutes=20),periods=39,freq='10min');hour=times.hour.to_numpy()+times.minute.to_numpy()/60
        future['hour_sin']=np.sin(2*np.pi*hour/24).astype(np.float32);future['hour_cos']=np.cos(2*np.pi*hour/24).astype(np.float32)
        # Explicit availability flags for supplied future plan, not true future observations.
        for k,pk in [(0,3),(5,4),(6,4)]:future[columns[k]+'_mask']=np.r_[np.repeat(masks[issue,k],3),np.repeat(plan[:,pk],3)].astype(np.float32)
        uni.append(target);cov.append(dict(target=target,past_covariates=pc,future_covariates=future))
    return uni,cov,empty

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['classical','foundation'],required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args();os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');torch.set_num_threads(4);a.out.mkdir(parents=True,exist_ok=False);lock=(a.out/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic();d=data(a.root);metrics=[];costs=[];old=json.loads((a.root/'plan_response_full_v2_20261019/protocol.json').read_text());dump(a.out/'protocol.json',dict(phase=a.phase,pilot=a.pilot,source_sha256=old['source_sha256'],training_indices_sha256=hashlib.sha256(d['train'].tobytes()).hexdigest(),history_steps10min=36,prediction_native10min=39,discard_gap10min=3,output_bins30min=12,all_eval_issues=True,future_actual=False,online_learning=False,independent_holdout=False,model_download=False))
    def save(name,seed,split,pred,seconds,**meta):
        ids=d['splits'][split];assert pred.shape==(len(ids),12) and np.isfinite(pred).all();neg=int((pred<0).sum());pred=np.maximum(pred,0).astype(np.float32);np.savez_compressed(a.out/(split+'__'+name+'__seed'+str(seed)+'.npz'),prediction=pred,target=d['target'][ids],issues=ids);metrics.extend(dict(seed=seed,name=name,split=split,**r) for r in measures(d,ids,pred));costs.append(dict(name=name,seed=seed,split=split,issues=len(ids),seconds=seconds,negative_bins_clipped=neg,**meta));pd.DataFrame(metrics).to_csv(a.out/'metrics.csv',index=False);pd.DataFrame(costs).to_csv(a.out/'costs.csv',index=False)
    if a.phase=='classical':
        x=tabular_features(d);pos=d['index'][d['train']];valid=np.isfinite(d['target'][d['train']]);design=x[pos][valid];y=d['target'][d['train']][valid];recent=[]
        for issue in d['scope']:
            ii=issue-np.arange(6);mask=d['mask'][ii,7]>0;v=d['received'][ii,7][mask];recent.append(float(np.mean(v)*3) if len(v) else np.nan)
        fallback=float(np.nanmedian(y));recent=np.where(np.isfinite(recent),recent,fallback)
        for split in ['validation','test']:
            ids=d['splits'][split];pred=np.repeat(recent[d['index'][ids],None],12,1);save('최근관측지속',0,split,pred,0.,training_seconds=0.,deterministic=True)
        for kind in ['Ridge','RandomForest','HistGradientBoosting']:
            for seed in ([13,37,71] if kind=='RandomForest' else [0]):
                if kind=='Ridge':m=Ridge(alpha=1.)
                elif kind=='RandomForest':m=RandomForestRegressor(n_estimators=64,max_depth=12,min_samples_leaf=5,n_jobs=4,random_state=seed)
                else:m=HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,learning_rate=.05,l2_regularization=.1,early_stopping=False,random_state=0)
                tick=time.monotonic();m.fit(design,y);elapsed=time.monotonic()-tick;assert elapsed<21600;joblib.dump(m,a.out/(kind+'__seed'+str(seed)+'.joblib'))
                serial=[]
                for at in d['index'][d['splits']['validation'][:32]]:
                    t=time.monotonic();m.predict(x[at]);serial.append(time.monotonic()-t)
                dump(a.out/(kind+'__seed'+str(seed)+'_serial.json'),dict(n=32,median_seconds=float(np.median(serial)),p95_seconds=float(np.quantile(serial,.95)),feature_preprocessing_excluded=True))
                for split in ['validation','test']:
                    ids=d['splits'][split];xx=x[d['index'][ids]];t=time.monotonic();pred=m.predict(xx.reshape(-1,xx.shape[-1])).reshape(-1,12);save(kind,seed,split,pred,time.monotonic()-t,training_seconds=elapsed,deterministic=kind!='RandomForest')
                print('CLASSICAL COMPLETE',kind,seed,elapsed,flush=True)
    else:
        from chronos import Chronos2Pipeline
        allids=np.r_[d['splits']['validation'],d['splits']['test']];trainfuel=d['received'][d['train'],7];median=float(np.nanmedian(trainfuel));uni,cov,empty=foundation_inputs(d,allids,median);dump(a.out/'input_audit.json',dict(target_all_missing_fallback_issues=empty,input_only_fallback_median=median,GT_unchanged=True,native_history36=True,cross_learning=False,future_actual_not_read=True))
        t=time.monotonic();pipe=Chronos2Pipeline.from_pretrained('amazon/chronos-2',device_map='cpu',dtype=torch.float32,local_files_only=True);cold=time.monotonic()-t;files=sorted(Path('/root/.cache/huggingface/hub/models--amazon--chronos-2/snapshots').rglob('*.safetensors'));dump(a.out/'chronos_model_sha.json',{str(f):sha(f) for f in files});limit=4 if a.pilot else len(allids)
        for label,inputs in [('Chronos2단독_연료',uni),('Chronos2단독_공변량계획',cov)]:
            chunks=[];beg=time.monotonic();one=[]
            for left in range(0,limit,8):
                t=time.monotonic()
                with torch.inference_mode():qq,_=pipe.predict_quantiles(inputs[left:min(left+8,limit)],prediction_length=39,context_length=36,quantile_levels=[.5],batch_size=64,cross_learning=False)
                z=np.stack([v[0,:,0].float().numpy() for v in qq]);assert z.shape[1]==39 and np.isfinite(z).all();chunks.append(aggregate_native10(z));one.append(time.monotonic()-t)
                if left%128==0:print('FOUNDATION',label,left+len(z),limit,flush=True)
            pred=np.concatenate(chunks);elapsed=time.monotonic()-beg
            if a.pilot:
                estimate=elapsed/limit*len(allids);assert estimate<21600;dump(a.out/(label+'_pilot.json'),dict(passed=True,seconds=elapsed,estimated_all_seconds=estimate,shape=list(pred.shape),cold_load_seconds=cold));continue
            for split in ['validation','test']:
                sl=slice(0,len(d['splits']['validation'])) if split=='validation' else slice(len(d['splits']['validation']),None);save(label,0,split,pred[sl],elapsed*len(d['splits'][split])/len(allids),cold_load_seconds=cold,covariates=label.endswith('公변량계획'.replace('公','공')),cross_learning=False)
        # Serial native comparison and proposed cache generation: do not compare cache-only with full FM without disclosure.
        from operational_core import causal_channels
        blocks=causal_channels(d['f'],3);serial=[]
        for issue in allids[:8]:
            t=time.monotonic()
            with torch.inference_mode():pipe.predict_quantiles([cov[np.flatnonzero(allids==issue)[0]]],prediction_length=39,context_length=36,quantile_levels=[.5],batch_size=64,cross_learning=False)
            native=time.monotonic()-t;inp=[]
            for values,mask,age in blocks:
                v=values[issue-11:issue+1].astype(np.float32).T.copy()
                if not np.isfinite(v).any():v[0,-1]=0.
                inp.append(v)
            t=time.monotonic()
            with torch.inference_mode():pipe.embed(inp,batch_size=32,context_length=12)
            serial.append(dict(issue=int(issue),native_covariate_seconds=native,dual_cache_generation_seconds=time.monotonic()-t))
        pd.DataFrame(serial).to_csv(a.out/'foundation_serial_and_cache_cost.csv',index=False);del pipe;gc.collect()
        import timesfm
        t=time.monotonic();model=timesfm.TimesFM_2p5_200M_torch.from_pretrained('google/timesfm-2.5-200m-pytorch',local_files_only=True);model.compile(timesfm.ForecastConfig(max_context=64,max_horizon=64,normalize_inputs=True,use_continuous_quantile_head=True,force_flip_invariance=True,infer_is_positive=True,fix_quantile_crossing=True,return_backcast=False,per_core_batch_size=4));cold=time.monotonic()-t;inp=[np.where(np.isfinite(v),v,median).astype(np.float32) for v in uni];missing=sum(int((~np.isfinite(v)).sum()) for v in uni);chunks=[];beg=time.monotonic()
        for left in range(0,limit,32):
            with torch.inference_mode():pred=model.forecast(horizon=39,inputs=inp[left:min(left+32,limit)])[0]
            assert np.isfinite(pred).all();chunks.append(aggregate_native10(pred))
        pred=np.concatenate(chunks);elapsed=time.monotonic()-beg
        if a.pilot:assert elapsed/limit*len(allids)<21600;dump(a.out/'TimesFM_pilot.json',dict(passed=True,estimated_all_seconds=elapsed/limit*len(allids)));print('FOUNDATION PILOT PASS',flush=True);return
        serial=[]
        for v in inp[:8]:
            t=time.monotonic()
            with torch.inference_mode():model.forecast(horizon=39,inputs=[v])
            serial.append(time.monotonic()-t)
        dump(a.out/'timesfm_serial.json',dict(n=8,median_seconds=float(np.median(serial)),p95_seconds=float(np.quantile(serial,.95)),preprocessing_excluded=True));files=sorted(p for p in Path('/root/.cache/huggingface/hub/models--google--timesfm-2.5-200m-pytorch/snapshots').rglob('*') if p.suffix in ['.safetensors','.bin','.pth','.pt']);dump(a.out/'timesfm_model_sha.json',{str(f):sha(f) for f in files})
        for split in ['validation','test']:
            sl=slice(0,len(d['splits']['validation'])) if split=='validation' else slice(len(d['splits']['validation']),None);save('TimesFM2.5단독_연료',0,split,pred[sl],elapsed*len(d['splits'][split])/len(allids),cold_load_seconds=cold,input_missing_filled_train_median=missing,covariates=False)
    dump(a.out/'completion.json',dict(complete=True,phase=a.phase,seconds=time.monotonic()-start,all_eval_issues=True,independent_holdout=False,model_download=False));dump(a.out/'run_sha256.json',{str(p.relative_to(a.out)):sha(p) for p in a.out.rglob('*') if p.is_file() and p.name!='lock'});print('COMPLETE BASELINES',a.phase,flush=True)
if __name__=='__main__':main()
