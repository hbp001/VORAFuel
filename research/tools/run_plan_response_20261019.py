"""Matched causal-plan comparison and separately labelled conditional-teacher study."""
import argparse,copy,fcntl,json,os,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from operational_core import Policy,causal_channels,score
from replay_grpo_base_20261015 import BASE,SOURCE_SHA,CACHE_SHA,SCALER_SHA,CHECKPOINT_SHA,features,sha,state_sha,CountAdam
from relational_ar_20261016 import ARCHS,SCALE,SixHourPolicy,histories,six_hour_targets,issue_split
from planned_multiscale_20261015 import MODES,plan_tokens
from online_ar_repair_20261016 import state_weights
from plan_response_20261019 import VARIANTS,PlanResponsePolicy,learning_loss,conditioned_grpo

TRACKS=['인과계획학습','혼합조건학습진단'];INFO=['발행가용모의계획','실제미래운항oracle진단']
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))

def data(root):
    source=root/'operational_qc_v3_regenerated_20261013/merged_qc_frame.csv.gz';cache=root/'fresh_fm_cache_20261013/fresh_dual_cache.npz';base=root/BASE
    assert sha(source)==SOURCE_SHA and sha(cache)==CACHE_SHA and sha(base/'scalers.pt')==SCALER_SHA
    f=pd.read_csv(source,parse_dates=['timestamp']).set_index('timestamp');target,release=six_hour_targets(f);splits={s:issue_split(f,s) for s in ['train','validation','test']}
    c=splits['train'][np.isfinite(target[splits['train']]).any(1)];train=c[np.linspace(0,len(c)-1,4096,dtype=int)];assert release[train].max()<=splits['validation'][0]
    scope=np.unique(np.r_[train,splits['validation'],splits['test']]);index=np.full(len(f),-1,np.int64);index[scope]=np.arange(len(scope));scaler=torch.load(base/'scalers.pt',weights_only=False)
    history=torch.tensor(histories(f,scaler)[scope]);x=features(f,scaler,np.load(cache));fm=x[2][scope];del x
    (v,m,_),(w,wm,_)=causal_channels(f,3)
    raw=np.column_stack([v[:,0]/50,v[:,7]*10,v[:,5],v[:,6],w[:,2]/10,w[:,3]/10,w[:,0]/3,w[:,4],w[:,5],m[:,0],m[:,7],m[:,5]*m[:,6],wm[:,2]*wm[:,3],wm[:,0]*wm[:,4]*wm[:,5]])
    raw=torch.tensor(np.nan_to_num(raw[scope]).astype(np.float32))
    plans=[torch.tensor(plan_tokens(f,scope,mode)[:,1:13]) for mode in [MODES[1],MODES[2]]]
    weights=torch.tensor(state_weights(v[scope,0],m[scope,0]));truth=torch.tensor(target[scope]/SCALE)
    return dict(f=f,target=target,release=release,splits=splits,train=train,scope=scope,index=index,history=history,fm=fm,raw=raw,plans=plans,weights=weights,truth=truth,received=v,mask=m)

def backbone(root,seed,nf):
    torch.manual_seed(seed);base=root/BASE;legacy=Policy(288,216,nf);legacy.load_state_dict(torch.load(base/f'checkpoint_seed{seed}.pt',weights_only=False)['grpo'])
    model=SixHourPolicy(legacy,ARCHS[2]);cp=root/'online_ar_repair_20261016'/f'seed{seed}'/'총량저속보상.pt';loaded=torch.load(cp,weights_only=False)
    assert loaded['protocol']['source_sha256']==SOURCE_SHA and loaded['protocol']['model_code_sha256']==sha(Path(__file__).resolve().parents[1]/'src/relational_ar_20261016.py')
    model.load_state_dict(loaded['state']);model.decoder_only();model.eval();return model,sha(cp)

def encode(model,d):
    chunks=[]
    with torch.no_grad():
        for left in range(0,len(d['scope']),128):chunks.append(model.encode(d['history'][left:left+128],d['fm'][left:left+128]))
    return tuple(torch.cat([c[j] for c in chunks]) for j in range(2))+(d['raw'],torch.zeros(len(d['scope']),1))

def predict(model,e,plans,pos):
    pieces=[]
    with torch.inference_mode():
        for left in range(0,len(pos),128):
            at=pos[left:left+128];pieces.append(model.predict_encoded(tuple(x[at] for x in e),plans[at]).numpy()*SCALE)
    return np.concatenate(pieces)

def rows(d,pred,ids,metadata):
    v=d['received'];m=d['mask'];gt=d['target'][ids];out=[]
    for state,mask in [('전체',np.ones(len(ids),bool)),('준정지속도대리',(v[ids,0]<=.5)&(m[ids,0]>0)),('저속대리',(v[ids,0]>.5)&(v[ids,0]<=3)&(m[ids,0]>0)),('항행속도대리',(v[ids,0]>3)&(m[ids,0]>0))]:
        for segment,sl in [('전체',slice(None)),('첫1시간',slice(0,2)),('3~6시간',slice(6,12))]:out.append(dict(**metadata,state=state,segment=segment,units='MT/30min',**score(gt[mask,sl].ravel(),pred[mask,sl].ravel())))
        full=mask&np.isfinite(gt).all(1);out.append(dict(**metadata,state=state,segment='완전6시간총량',units='MT/6h',**score(gt[full].sum(1),pred[full].sum(1))))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    assert not os.environ.get('CUDA_VISIBLE_DEVICES','');torch.set_num_threads(4);a.out.mkdir(parents=True,exist_ok=False)
    lock=(a.out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic();d=data(a.root);tp=d['index'][d['train']]
    protocol=dict(source_sha256=SOURCE_SHA,cache_sha256=CACHE_SHA,scaler_sha256=SCALER_SHA,model_code_sha256=sha(Path(__file__).resolve().parents[1]/'src/plan_response_20261019.py'),runner_sha256=sha(Path(__file__)),training_indices_sha256=__import__('hashlib').sha256(d['train'].tobytes()).hexdigest(),warm_steps=256,warm_lr=3e-4,post_steps=40,post_lr=3e-5,batch=64,tracks=TRACKS,variants=VARIANTS,contexts=36,bins=12,bin_minutes=30,source_flag='0 historical synthetic / 1 supplied-condition diagnostic',causal_vs_oracle_separated=True,ensemble=False,automatic_deployment=False,manuscript_written=False,independent_holdout=False,pilot=a.pilot)
    assert protocol['training_indices_sha256']==json.loads((a.root/'online_ar_repair_20261016/protocol.json').read_text())['training_indices_sha256'];write(a.out/'protocol.json',protocol)
    metrics=[];training=[]
    for seed in ([13] if a.pilot else [13,37,71]):
        base,parent_sha=backbone(a.root,seed,d['fm'].shape[1]);e=encode(base,d);dest=a.out/f'seed{seed}';dest.mkdir();write(dest/'parent.json',dict(sha256=parent_sha,state_sha256=state_sha(base.state_dict())))
        initial=PlanResponsePolicy(base,VARIANTS[0]);ids=d['splits']['validation'];pos=d['index'][ids]
        expected=np.load(a.root/'online_ar_repair_20261016'/f'seed{seed}'/'validation_총량저속보상.npz')
        np.testing.assert_allclose(predict(initial,e,d['plans'][0],pos),expected['prediction'],atol=1e-6,rtol=0)
        if a.pilot:
            m=PlanResponsePolicy(base,VARIANTS[3]);opt=CountAdam([p for p in m.parameters() if p.requires_grad],lr=3e-4);at=tp[:64];batch=tuple(x[at] for x in e);times=[]
            for k in range(3):
                tick=time.monotonic();loss=learning_loss(m,batch,d['plans'][0][at],d['truth'][at],d['weights'][at]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.);opt.step();times.append(time.monotonic()-tick)
            reference=copy.deepcopy(m);t=time.monotonic();conditioned_grpo(m,reference,batch,d['plans'][0][at],d['truth'][at],opt,d['weights'][at]);pg=time.monotonic()-t
            estimate=float(np.median(times[1:]))*256+pg*20;assert estimate<21600 and all(torch.isfinite(p).all() for p in m.parameters());write(a.out/'pilot.json',dict(passed=True,warm_step_seconds=float(np.median(times[1:])),grpo_two_steps_seconds=pg,estimated_fit_seconds=estimate));print('PILOT PASS',estimate,flush=True);return
        for split in ['validation','test']:
            ids=d['splits'][split];pred=predict(initial,e,d['plans'][0],d['index'][ids]);metrics.extend(rows(d,pred,ids,dict(seed=seed,track='기존보완고정',variant='기존보완고정',arm='GRPO고정',information=INFO[0],split=split)))
        for track in TRACKS:
            for variant in (VARIANTS if track==TRACKS[0] else [VARIANTS[0],VARIANTS[3]]):
                assert time.monotonic()-start<21600
                torch.manual_seed(seed+19000);model=PlanResponsePolicy(base,variant);rng=np.random.default_rng(seed+19100)
                batches=[rng.choice(tp,64,replace=False) for _ in range(276)];teacher=[torch.tensor(rng.random(64)<.5) if track==TRACKS[1] else torch.zeros(64,dtype=torch.bool) for _ in batches]
                def batch(k):
                    at=batches[k];flag=teacher[k];pe=d['plans'][0][at].clone();pe[flag]=d['plans'][1][at][flag]
                    assert np.isin(d['scope'][at],d['train']).all();assert (d['release'][d['scope'][at]].max()<=d['splits']['validation'][0])
                    return tuple(x[at] for x in e[:3])+(flag.float()[:,None],),pe,d['truth'][at],d['weights'][at]
                opt=CountAdam([p for p in model.parameters() if p.requires_grad],lr=3e-4);tick=time.monotonic()
                for k in range(256):
                    be,plans,y,w=batch(k);loss=learning_loss(model,be,plans,y,w);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step()
                    assert torch.isfinite(loss)
                assert opt.actual_steps==256;warm=copy.deepcopy(model);reference=copy.deepcopy(model);sft=copy.deepcopy(model)
                og=CountAdam([p for p in model.parameters() if p.requires_grad],lr=3e-5);osft=CountAdam([p for p in sft.parameters() if p.requires_grad],lr=3e-5)
                torch.manual_seed(seed+19200)
                for k in range(256,276):
                    be,plans,y,w=batch(k);conditioned_grpo(model,reference,be,plans,y,og,w)
                    if variant==VARIANTS[3]:
                        for _ in range(2):
                            loss=learning_loss(sft,be,plans,y,w);osft.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in sft.parameters() if p.requires_grad],1.);osft.step()
                assert og.actual_steps==40
                for arm,active in [('GRPO고정',model)]+([('동일목적SFT고정',sft)] if variant==VARIANTS[3] else []):
                    assert all(torch.isfinite(p).all() for p in active.parameters())
                    for key,v in base.state_dict().items():
                        if key.startswith(('vessel.','weather.','fm.','relation.')):assert torch.equal(v,active.backbone.state_dict()[key])
                    name=track+'__'+variant+'__'+arm;path=dest/(name+'.pt');torch.save(dict(state=active.state_dict(),warm=warm.state_dict(),protocol=protocol,parent_sha256=parent_sha,track=track,variant=variant,arm=arm),path)
                    # Only validation source0 is evaluated before the frozen choice.
                    ids=d['splits']['validation'];pred=predict(active,e,d['plans'][0],d['index'][ids]);np.savez_compressed(dest/('validation__'+name+'__'+INFO[0]+'.npz'),prediction=pred,target=d['target'][ids],issues=ids)
                    metrics.extend(rows(d,pred,ids,dict(seed=seed,track=track,variant=variant,arm=arm,information=INFO[0],split='validation')))
                training.append(dict(seed=seed,track=track,variant=variant,warm_steps=opt.actual_steps,grpo_steps=og.actual_steps,extra_sft_control_steps=osft.actual_steps,seconds=time.monotonic()-tick,parameters=sum(p.numel() for p in model.parameters())))
                print(json.dumps(dict(stage='validation_fit',**training[-1]),ensure_ascii=False),flush=True)
                pd.DataFrame(metrics).to_csv(a.out/'metrics.csv',index=False);pd.DataFrame(training).to_csv(a.out/'training.csv',index=False)
        del e
    df=pd.DataFrame(metrics)
    def component(track,variant):
        part=df[(df.track==track)&(df.variant==variant)&(df.arm=='GRPO고정')&(df.split=='validation')]
        def metric(state,seg):return float(part[(part.state==state)&(part.segment==seg)].mae.mean())
        return dict(point=metric('전체','전체'),total=metric('전체','완전6시간총량'),slow=metric('준정지속도대리','전체'))
    baseline=component('기존보완고정','기존보완고정');candidates={v:component(TRACKS[0],v) for v in VARIANTS};eligible=[]
    for j,v in enumerate(VARIANTS):
        c=candidates[v]
        if c['point']<=1.03*baseline['point'] and c['total']<=1.03*baseline['total'] and c['slow']<=baseline['slow']:
            eligible.append((c['point']/baseline['point']+.5*c['total']/baseline['total']+c['slow']/baseline['slow'],j,v))
    selected=min(eligible)[2] if eligible else None;write(a.out/'selection.json',dict(baseline=baseline,candidates=candidates,selected=selected,test_used=False,oracle_used=False));print('VALIDATION SELECTED '+str(selected),flush=True)
    for seed in [13,37,71]:
        base,_=backbone(a.root,seed,d['fm'].shape[1]);e=encode(base,d);dest=a.out/f'seed{seed}'
        for path in dest.glob('*.pt'):
            cp=torch.load(path,weights_only=False);model=PlanResponsePolicy(base,cp['variant']);model.load_state_dict(cp['state']);model.eval();name=cp['track']+'__'+cp['variant']+'__'+cp['arm']
            for split in ['validation','test']:
                for info in [0,1]:
                    if split=='validation' and info==0:continue
                    ids=d['splits'][split];view=e[:3]+(torch.full_like(e[3],float(info)),);pred=predict(model,view,d['plans'][info],d['index'][ids]);assert np.isfinite(pred).all() and (pred>=0).all()
                    np.savez_compressed(dest/(split+'__'+name+'__'+INFO[info]+'.npz'),prediction=pred,target=d['target'][ids],issues=ids)
                    metrics.extend(rows(d,pred,ids,dict(seed=seed,track=cp['track'],variant=cp['variant'],arm=cp['arm'],information=INFO[info],split=split)))
        pd.DataFrame(metrics).to_csv(a.out/'metrics.csv',index=False)
    write(a.out/'completion.json',dict(complete=True,selected=selected,seeds=[13,37,71],seconds=time.monotonic()-start,independent_holdout=False,manuscript_written=False,automatic_deployment=False));print('COMPLETE plan response experiment',flush=True)

if __name__=='__main__':main()
