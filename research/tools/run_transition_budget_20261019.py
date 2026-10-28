"""Exploratory joint mass/allocation continuation; prior tests are already seen."""
import argparse,copy,fcntl,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_plan_response_20261019 import data,backbone,encode,rows,TRACKS
from plan_response_20261019 import PlanResponsePolicy,VARIANTS as PREVIOUS
from transition_budget_20261019 import TransitionPolicy,VARIANTS,loss as training_loss,update as grpo
from replay_grpo_base_20261015 import BASE,sha,CountAdam,state_sha

def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def recent_info(d,root):
    v=np.stack([d['received'][d['scope']-k,7] for k in range(6)],1);fresh=np.stack([d['mask'][d['scope']-k,7]>0 for k in range(6)],1)
    out=[]
    for j in range(len(v)):
        ok=fresh[j];x=v[j,ok];rate=float(np.mean(x))*10 if len(x) else float(d['raw'][j,1]);new=v[j,:3][ok[:3]];old=v[j,3:][ok[3:]]
        trend=float(np.mean(new)-np.mean(old))*10 if len(new) and len(old) else 0.
        out.append([max(0,rate),float(ok.mean()),trend])
    return torch.tensor(out,dtype=torch.float32)
def parent(root,seed,nf):
    base,_=backbone(root,seed,nf);p=root/'plan_response_full_v2_20261019'/f'seed{seed}'/(TRACKS[0]+'__'+PREVIOUS[1]+'__GRPO고정.pt');cp=torch.load(p,weights_only=False)
    assert cp['protocol']['model_code_sha256']==sha(Path(__file__).resolve().parents[1]/'src/plan_response_20261019.py');m=PlanResponsePolicy(base,PREVIOUS[1]);m.load_state_dict(cp['state']);m.eval();m.decoder_only();return m,sha(p)
def inference(m,e,p,pos):
    chunks=[]
    with torch.inference_mode():
        for k in range(0,len(pos),128):
            at=pos[k:k+128];chunks.append(m.predict_encoded(tuple(x[at] for x in e),p[at]).numpy()*.3)
    return np.concatenate(chunks)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args();assert not os.environ.get('CUDA_VISIBLE_DEVICES','');torch.set_num_threads(4);a.out.mkdir(parents=True,exist_ok=False);lock=(a.out/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();d=data(a.root);recent=recent_info(d,a.root);tp=d['index'][d['train']];plans=d['plans'][0];metrics=[];training=[];oldproto=json.loads((a.root/'plan_response_full_v2_20261019/protocol.json').read_text());indexsha=hashlib.sha256(d['train'].tobytes()).hexdigest();assert indexsha==oldproto['training_indices_sha256']
    speeds=d['f'].SOG.where(d['f'].SOG.between(0,50)).to_numpy();future_speed=np.stack([np.stack([speeds[d['scope']+h-k] for k in [0,1,2]],1) for h in range(3,37,3)],1);known=np.isfinite(future_speed).all(2);aux_target=torch.tensor(np.where(known,(np.nan_to_num(future_speed).mean(2)<=.5).astype(float),np.nan),dtype=torch.float32);lowcells=(aux_target[tp].numpy()==1)&np.isfinite(d['target'][d['train']]);low_rate=float(np.median(d['target'][d['train']][lowcells])/.3)
    def IdleBudgetPolicy(par,variant):return TransitionPolicy(par,variant,low_rate)
    protocol=dict(source_sha256=oldproto['source_sha256'],cache_sha256=oldproto['cache_sha256'],scaler_sha256=oldproto['scaler_sha256'],training_indices_sha256=indexsha,core_sha256=sha(Path(__file__).resolve().parents[1]/'src/transition_budget_20261019.py'),runner_sha256=sha(Path(__file__)),previous_core_sha256=oldproto['model_code_sha256'],variants=VARIANTS,warm_steps=256,grpo_steps=40,batch=64,recent_fuel='mean genuinely fresh last6 ten-minute slots',train_only_low_rate_normalized=low_rate,train_low_rate_cells=int(lowcells.sum()),aux_train_known=int(torch.isfinite(aux_target[tp]).sum()),aux_ground_truth_in_forward=False,information='as-of past-day synthetic plan ONLY',horizon_bins=12,bin_minutes=30,latent_actions=12,exploratory_design_after_previous_test_seen=True,ensemble=False,future_GT_in_forward=False,independent_holdout=False,pilot=a.pilot,automatic_deployment=False,manuscript_written=False)
    write(a.out/'protocol.json',protocol)
    low=(d['received'][d['train'],0]<=.5)&(d['mask'][d['train'],0]>0);full=np.isfinite(d['target'][d['train']]).all(1)
    write(a.out/'train_coverage.json',dict(issues=4096,complete=int(full.sum()),nearstop=int(low.sum()),complete_nearstop=int((low&full).sum()),complete_fraction=float(full.mean())))
    for seed in ([13] if a.pilot else [13,37,71]):
        par,psha=parent(a.root,seed,d['fm'].shape[1]);e=encode(par.backbone,d)+(recent,);dest=a.out/f'seed{seed}';dest.mkdir();write(dest/'parent.json',dict(sha256=psha,state_sha256=state_sha(par.state_dict())))
        control=IdleBudgetPolicy(par,VARIANTS[0]);ids=d['splits']['validation'];expected=np.load(a.root/'plan_response_full_v2_20261019'/f'seed{seed}'/('validation__'+TRACKS[0]+'__'+PREVIOUS[1]+'__GRPO고정__발행가용모의계획.npz'));np.testing.assert_allclose(inference(control,e,plans,d['index'][ids]),expected['prediction'],atol=1e-6,rtol=0)
        if a.pilot:
            model=IdleBudgetPolicy(par,VARIANTS[-1]);at=tp[:64];be=tuple(x[at] for x in e);opt=CountAdam([p for p in model.parameters() if p.requires_grad],lr=3e-4);times=[]
            for _ in range(3):
                t=time.monotonic();loss=training_loss(model,be,plans[at],d['truth'][at],d['weights'][at],aux_target[at]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);opt.step();times.append(time.monotonic()-t)
            model.freeze_auxiliary();t=time.monotonic();grpo(model,copy.deepcopy(model),be,plans[at],d['truth'][at],opt,d['weights'][at]);seconds=time.monotonic()-t;estimate=float(np.median(times[1:]))*256+seconds*20;assert estimate<21600 and all(torch.isfinite(p).all() for p in model.parameters());write(a.out/'pilot.json',dict(passed=True,estimated_fit_seconds=estimate));print('PILOT PASS',estimate,flush=True);return
        for split in ['validation','test']:
            ids=d['splits'][split]
            for name,path in [('이전선택고정',a.root/'plan_response_full_v2_20261019'/f'seed{seed}'/(split+'__'+TRACKS[0]+'__'+PREVIOUS[1]+'__GRPO고정__발행가용모의계획.npz')),('최초보완고정',a.root/'online_ar_repair_20261016'/f'seed{seed}'/(split+'_총량저속보상.npz'))]:
                z=np.load(path);np.testing.assert_array_equal(z['issues'],ids);np.testing.assert_allclose(z['target'],d['target'][ids],equal_nan=True);np.savez_compressed(dest/(split+'__'+name+'.npz'),**{k:z[k] for k in ['prediction','target','issues']});metrics.extend(rows(d,z['prediction'],ids,dict(name=name,seed=seed,split=split)))
            rate=recent[d['index'][ids],0].numpy()*.3;pred=np.repeat(rate[:,None],12,1);np.savez_compressed(dest/(split+'__최근연료유지참고.npz'),prediction=pred,target=d['target'][ids],issues=ids);metrics.extend(rows(d,pred,ids,dict(name='최근연료유지참고',seed=seed,split=split)))
        for variant in VARIANTS:
            assert time.monotonic()-start<21600;torch.manual_seed(seed+29000);model=IdleBudgetPolicy(par,variant);rng=np.random.default_rng(seed+29100);batches=[rng.choice(tp,64,replace=False) for _ in range(276)];opt=CountAdam([p for p in model.parameters() if p.requires_grad],lr=3e-4);t=time.monotonic()
            def batch(k):
                at=batches[k];assert (d['release'][d['scope'][at]].max()<=d['splits']['validation'][0]);return tuple(x[at] for x in e),plans[at],d['truth'][at],d['weights'][at]
            for k in range(256):
                be,pp,y,w=batch(k);loss=training_loss(model,be,pp,y,w,aux_target[batches[k]]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);opt.step();assert torch.isfinite(loss)
            assert opt.actual_steps==256;model.freeze_auxiliary();warm=copy.deepcopy(model.state_dict());ref=copy.deepcopy(model);sft=copy.deepcopy(model);og=CountAdam([p for p in model.parameters() if p.requires_grad],lr=3e-5);osft=CountAdam([p for p in sft.parameters() if p.requires_grad],lr=3e-5);torch.manual_seed(seed+29200)
            for k in range(256,276):
                be,pp,y,w=batch(k);grpo(model,ref,be,pp,y,og,w)
                if variant==VARIANTS[-1]:
                    for _ in range(2):
                        loss=training_loss(sft,be,pp,y,w);osft.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in sft.parameters() if p.requires_grad],1);osft.step()
            assert og.actual_steps==40
            for arm,active in [('GRPO고정',model)]+([('동일목적SFT고정',sft)] if variant==VARIANTS[-1] else []):
                assert all(torch.isfinite(p).all() for p in active.parameters())
                for k,v in par.state_dict().items():
                    if k.startswith(('backbone.vessel.','backbone.weather.','backbone.fm.','backbone.relation.')):assert torch.equal(active.parent.state_dict()[k],v)
                for k,v in ref.aux.state_dict().items():assert torch.equal(v,active.aux.state_dict()[k])
                name=variant+'__'+arm;torch.save(dict(state=active.state_dict(),warm=warm,protocol=protocol,variant=variant,arm=arm,parent_sha256=psha),dest/(name+'.pt'))
                ids=d['splits']['validation'];pred=inference(active,e,plans,d['index'][ids]);np.savez_compressed(dest/('validation__'+name+'.npz'),prediction=pred,target=d['target'][ids],issues=ids);metrics.extend(rows(d,pred,ids,dict(seed=seed,name=name,split='validation')))
            training.append(dict(seed=seed,variant=variant,warm_steps=256,grpo_steps=40,SFT_control_steps=osft.actual_steps,seconds=time.monotonic()-t,parameters=sum(p.numel() for p in model.parameters())));pd.DataFrame(metrics).to_csv(a.out/'metrics.csv',index=False);pd.DataFrame(training).to_csv(a.out/'training.csv',index=False);print(json.dumps(training[-1],ensure_ascii=False),flush=True)
    df=pd.DataFrame(metrics)
    def metric(name,state,segment,key='mae'):
        return float(df[(df.name==name)&(df.split=='validation')&(df.state==state)&(df.segment==segment)][key].mean())
    base=dict(point=metric('이전선택고정','전체','전체'),slow=metric('이전선택고정','준정지속도대리','전체'),total=metric('이전선택고정','전체','완전6시간총량'),p95=metric('이전선택고정','전체','전체','p95'));candidates={};eligible=[]
    for j,v in enumerate(VARIANTS):
        name=v+'__GRPO고정';c=dict(point=metric(name,'전체','전체'),slow=metric(name,'준정지속도대리','전체'),total=metric(name,'전체','완전6시간총량'),p95=metric(name,'전체','전체','p95'));candidates[v]=c
        if c['point']<=base['point']*1.02 and c['slow']<=base['slow'] and c['total']<base['total'] and c['p95']<=base['p95']*1.05:eligible.append((.5*c['point']/base['point']+c['total']/base['total']+.5*c['slow']/base['slow'],j,v))
    selected=min(eligible)[2] if eligible else None;write(a.out/'selection.json',dict(selected=selected,baseline=base,candidates=candidates,test_used=False,oracle_used=False));print('VALIDATION SELECTED',selected,flush=True)
    for seed in [13,37,71]:
        par,_=parent(a.root,seed,d['fm'].shape[1]);e=encode(par.backbone,d)+(recent,);dest=a.out/f'seed{seed}';ids=d['splits']['test']
        for cpfile in dest.glob('*.pt'):
            cp=torch.load(cpfile,weights_only=False);m=IdleBudgetPolicy(par,cp['variant']);m.load_state_dict(cp['state']);m.eval();name=cp['variant']+'__'+cp['arm'];pred=inference(m,e,plans,d['index'][ids]);assert np.isfinite(pred).all() and (pred>=0).all();np.savez_compressed(dest/('test__'+name+'.npz'),prediction=pred,target=d['target'][ids],issues=ids);metrics.extend(rows(d,pred,ids,dict(seed=seed,name=name,split='test')))
        pd.DataFrame(metrics).to_csv(a.out/'metrics.csv',index=False)
    write(a.out/'completion.json',dict(complete=True,selected=selected,seeds=[13,37,71],seconds=time.monotonic()-start,independent_holdout=False,automatic_deployment=False,manuscript_written=False));print('COMPLETE transition budget',flush=True)

if __name__=='__main__':main()
