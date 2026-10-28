"""Predeclared frozen-policy residual calibration; validation-only safety gate."""
import argparse,fcntl,hashlib,json,os,sys,time,copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from conservative_budget_20261019 import ConservativeResidual,features,loss
from run_transition_budget_20261019 import parent,recent_info,inference
from run_plan_response_20261019 import data,encode
from operational_core import score
from replay_grpo_base_20261015 import sha,state_sha

WEIGHTS=[0.,.5,2.,8.]
ALPHAS=[.125,.25,.5,1.]
NAMES=['30분전용잔차대조','보수적총량잔차','총량균형잔차','총량강조잔차']
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def cohorts(d,ids):
    speed=d['f'].SOG.where(d['f'].SOG.between(0,50)).to_numpy()
    future=np.stack([np.stack([speed[ids+h-k] for k in [0,1,2]],1) for h in range(3,37,3)],1)
    known=np.isfinite(future).all(2);avg=np.nan_to_num(future).mean(2)
    current=(d['mask'][ids,0]>0)&(d['received'][ids,0]<=.5)
    stay=current[:,None]&known&(avg<=.5);depart=current[:,None]&known&(avg>3)
    windows=dict(유지window=current&known.all(1)&(avg<=.5).all(1),전환window=current&known.all(1)&(avg[:,0]<=.5)&(avg>3).any(1))
    return current,stay,depart,windows
def measures(d,ids,pred):
    gt=d['target'][ids];current,stay,depart,windows=cohorts(d,ids)
    masks={'전체':np.ones_like(gt,dtype=bool),'현재준정지':np.repeat(current[:,None],12,1),'준정지유지':stay,'출항전환':depart,'첫30분':np.broadcast_to(np.arange(12)==0,gt.shape)}
    out=[dict(segment=k,units='MT/30min',**score(gt[m],pred[m])) for k,m in masks.items()]
    full=np.isfinite(gt).all(1)
    for k,mask in {'완전6시간':full,**{k:full&v for k,v in windows.items()}}.items():
        out.append(dict(segment=k,units='MT/6h',**score(gt[mask].sum(1),pred[mask].sum(1))))
    return out
def metrics_map(d,ids,pred):return {r['segment']:r for r in measures(d,ids,pred)}
def candidate_name(j,alpha):return NAMES[j]+'__강도'+str(alpha)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    assert not os.environ.get('CUDA_VISIBLE_DEVICES','');torch.set_num_threads(4);a.out.mkdir(parents=True,exist_ok=False);lock=(a.out/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();d=data(a.root);recent=recent_info(d,a.root);tp=d['index'][d['train']];vp=d['index'][d['splits']['validation']];plan=d['plans'][0];proto=json.loads((a.root/'plan_response_full_v2_20261019/protocol.json').read_text());ish=hashlib.sha256(d['train'].tobytes()).hexdigest();assert ish==proto['training_indices_sha256'];assert d['release'][d['train']].max()<=d['splits']['validation'][0]
    protocol=dict(source_sha256=proto['source_sha256'],cache_sha256=proto['cache_sha256'],scaler_sha256=proto['scaler_sha256'],training_indices_sha256=ish,code_sha256=sha(Path(__file__).resolve().parents[1]/'src/conservative_budget_20261019.py'),runner_sha256=sha(Path(__file__)),parent_all_parameters_frozen=True,trainable_parameters=12,updates=256,lr=.01,full_batch=4096,total_weights=WEIGHTS,validation_alphas=ALPHAS,GRPO_new_training=False,supervised_calibration_of_frozen_GRPO=True,independent_holdout=False,ensemble=False,future_actual_in_forward=False,test_used_for_selection=False,automatic_deployment=False,pilot=a.pilot)
    dump(a.out/'protocol.json',protocol);_,stay,depart,_=cohorts(d,d['train']);traincohorts=[torch.tensor(stay),torch.tensor(depart)];allmetrics=[];training=[];checkpoints=[]
    # New qualitative cases chosen from validation metadata before any fitting.
    vi=d['splits']['validation'];_,st,de,win=cohorts(d,vi);full=np.isfinite(d['target'][vi]).all(1);cases=[]
    for key,mask in [('준정지유지',win['유지window']),('출항전환',win['전환window'])]:
        valid=vi[mask&full];fallback=False
        if not len(valid):valid=vi[((st if key=='준정지유지' else de).any(1))&full];fallback=True
        cases.append(dict(cohort=key,issue=int(valid[0]) if len(valid) else None,time=str(d['f'].index[valid[0]]) if len(valid) else None,fallback_cell_cohort=fallback,chosen_by='chronological first complete validation, no prediction inspection'))
    dump(a.out/'predeclared_cases.json',cases)
    for seed in ([13] if a.pilot else [13,37,71]):
        par,psha=parent(a.root,seed,d['fm'].shape[1]);par.eval()
        for param in par.parameters():param.requires_grad_(False)
        initial_state=state_sha(par.state_dict());e=encode(par.backbone,d);base=torch.tensor(inference(par,e,plan,np.arange(len(d['scope']))));x=features(base,d['raw'],plan,recent).detach();dest=a.out/f'seed{seed}';dest.mkdir();dump(dest/'parent.json',dict(checkpoint_sha256=psha,state_sha256=initial_state))
        from run_plan_response_20261019 import TRACKS
        from plan_response_20261019 import VARIANTS as PARENT_VARIANTS
        expected=np.load(a.root/'plan_response_full_v2_20261019'/f'seed{seed}'/('validation__'+TRACKS[0]+'__'+PARENT_VARIANTS[1]+'__GRPO고정__발행가용모의계획.npz'))
        np.testing.assert_array_equal(expected['issues'],d['splits']['validation']);np.testing.assert_allclose(base[vp].numpy(),expected['prediction'],atol=1e-6,rtol=0)
        for split in ['validation','test']:
            ids=d['splits'][split];pos=d['index'][ids];np.savez_compressed(dest/(split+'__기존고정.npz'),prediction=base[pos].numpy(),target=d['target'][ids],issues=ids)
        b=base[tp];xx=x[tp];target=d['truth'][tp]*.3
        if a.pilot:
            m=ConservativeResidual();assert torch.equal(m(b,xx),b);opt=torch.optim.Adam(m.parameters(),lr=.01);t=time.monotonic()
            for _ in range(3):
                objective=loss(m,b,xx,target,traincohorts,2);opt.zero_grad();objective.backward();opt.step();assert torch.isfinite(objective)
            estimate=(time.monotonic()-t)/3*256;assert estimate<21600
            assert state_sha(par.state_dict())==initial_state
            dump(a.out/'pilot.json',dict(passed=True,estimated_fit_seconds=estimate,parent_frozen=True,zero_initial_bitexact=True));print('PILOT PASS',estimate,flush=True);return
        for j,tw in enumerate(WEIGHTS):
            torch.manual_seed(seed);m=ConservativeResidual();opt=torch.optim.Adam(m.parameters(),lr=.01);t=time.monotonic();loss_trace=[]
            for k in range(256):
                objective=loss(m,b,xx,target,traincohorts,tw);opt.zero_grad();objective.backward();assert torch.isfinite(m.weight.grad).all();opt.step();loss_trace.append(float(objective));assert torch.isfinite(objective)
            assert time.monotonic()-t<21600;assert state_sha(par.state_dict())==initial_state
            torch.save(dict(state=m.state_dict(),parent_sha256=psha,protocol=protocol,total_weight=tw),dest/(NAMES[j]+'.pt'))
            with torch.inference_mode():
                for alpha in ALPHAS:
                    pred=m(base[vp],x[vp],alpha).numpy();ids=d['splits']['validation'];name=candidate_name(j,alpha);np.savez_compressed(dest/('validation__'+name+'.npz'),prediction=pred,target=d['target'][ids],issues=ids)
                    allmetrics.extend(dict(seed=seed,split='validation',name=name,**r) for r in measures(d,ids,pred))
            training.append(dict(seed=seed,name=NAMES[j],steps=256,seconds=time.monotonic()-t,first_loss=loss_trace[0],last_loss=loss_trace[-1],parent_frozen=True));print(json.dumps(training[-1],ensure_ascii=False),flush=True)
        for split in ['validation','test']:
            ids=d['splits'][split];pos=d['index'][ids];allmetrics.extend(dict(seed=seed,split=split,name='기존고정',**r) for r in measures(d,ids,base[pos].numpy()))
    frame=pd.DataFrame(allmetrics);val=frame[frame.split=='validation'];mean=val.groupby(['name','segment'])[['mae','rmse','p95']].mean();base_metrics=mean.loc['기존고정'];gates=[];eligible=[]
    for j,tw in enumerate(WEIGHTS):
        for alpha in ALPHAS:
            name=candidate_name(j,alpha);v=mean.loc[name];checks={}
            for segment,keys in [('전체',['mae','rmse','p95']),('현재준정지',['mae']),('준정지유지',['mae']),('출항전환',['mae'])]:
                for key in keys:checks[segment+'_'+key]=bool(np.isfinite(v.loc[segment,key]) and v.loc[segment,key]<=base_metrics.loc[segment,key]+1e-8)
            checks['total_improves']=bool(v.loc['완전6시간','mae']<base_metrics.loc['완전6시간','mae'])
            for seed in [13,37,71]:
                part=val[(val.seed==seed)&(val.segment=='전체')].set_index('name');checks['seed'+str(seed)+'_MAE']=bool(part.loc[name,'mae']<=part.loc['기존고정','mae']*1.01)
            passed=all(checks.values());gates.append(dict(name=name,passed=passed,checks=checks,total_mae=float(v.loc['완전6시간','mae'])))
            if passed:eligible.append((float(v.loc['완전6시간','mae']),alpha,tw,j,name))
    selected=min(eligible)[-1] if eligible else '기존고정';selection=dict(selected=selected,eligible=len(eligible),test_used=False,oracle_used=False,gates=gates);dump(a.out/'selection.json',selection);print('VALIDATION SELECTED',selected,flush=True)
    # Do not evaluate the whole strength grid on test; only frozen choice + two predeclared contrasts.
    final_names=list(dict.fromkeys([selected,candidate_name(0,1.),candidate_name(2,1.)]));audits=[]
    for seed in [13,37,71]:
        par,psha=parent(a.root,seed,d['fm'].shape[1]);par.eval();state_before=state_sha(par.state_dict());e=encode(par.backbone,d);base=torch.tensor(inference(par,e,plan,np.arange(len(d['scope']))));x=features(base,d['raw'],plan,recent).detach();dest=a.out/f'seed{seed}'
        for name in final_names:
            if name=='기존고정':continue
            j=next(j for j in range(4) if name.startswith(NAMES[j]+'__'));alpha=float(name.split('__강도')[1]);cp=torch.load(dest/(NAMES[j]+'.pt'),weights_only=False);assert cp['parent_sha256']==psha;m=ConservativeResidual();m.load_state_dict(cp['state']);m.eval()
            ids=d['splits']['test'];pos=d['index'][ids]
            with torch.inference_mode():pred=m(base[pos],x[pos],alpha).numpy()
            assert np.isfinite(pred).all() and (pred>=0).all();np.savez_compressed(dest/('test__'+name+'.npz'),prediction=pred,target=d['target'][ids],issues=ids);allmetrics.extend(dict(seed=seed,split='test',name=name,**r) for r in measures(d,ids,pred))
        for j in range(4):
            cp=torch.load(dest/(NAMES[j]+'.pt'),weights_only=False);m=ConservativeResidual();m.load_state_dict(cp['state'])
            pos=vp[:8]
            with torch.inference_mode():pred=m(base[pos],x[pos],1.).numpy()
            z=np.load(dest/('validation__'+candidate_name(j,1.)+'.npz'));difference=float(abs(pred-z['prediction'][:8]).max());assert difference<1e-6
            audits.append(dict(seed=seed,name=NAMES[j],checkpoint_sha256=sha(dest/(NAMES[j]+'.pt')),parent_state_unchanged=state_before==state_sha(par.state_dict()),fresh_full_parent_inference=True,saved_prediction_max_abs_difference=difference))
    pd.DataFrame(allmetrics).to_csv(a.out/'metrics.csv',index=False);pd.DataFrame(training).to_csv(a.out/'training.csv',index=False);dump(a.out/'checkpoint_audit.json',dict(passed=True,checkpoints=len(audits),checks=audits));dump(a.out/'completion.json',dict(complete=True,selected=selected,seconds=time.monotonic()-start,seeds=[13,37,71],test_names=['기존고정']+[n for n in final_names if n!='기존고정'],automatic_deployment=False));dump(a.out/'run_sha256.json',{str(p.relative_to(a.out)):sha(p) for p in sorted(a.out.rglob('*')) if p.is_file() and p.name!='lock'});print('COMPLETE',flush=True)
if __name__=='__main__':main()
