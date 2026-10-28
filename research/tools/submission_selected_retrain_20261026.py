"""Frozen selected training chain from rebuilt inputs; no new model selection."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import sys,json,time,copy,fcntl,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
R=Path('/workspace');O=R/'submission_raw_20261026'
sys.path[:0]=[str(R/'condition_anchor_code_20261019/tools'),str(R/'condition_anchor_code_20261019/src')]
import run_plan_response_20261019 as run
from operational_core import Policy,SCALE,score
from relational_ar_20261016 import ARCHS,SixHourPolicy,masked_loss,grpo_update
from online_ar_repair_20261016 import grpo_repair
from plan_response_20261019 import VARIANTS,PlanResponsePolicy,learning_loss,conditioned_grpo
from replay_grpo_base_20261015 import BASE,CountAdam,sha,state_sha
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def main():
    dest=O/'selected_training';dest.mkdir(exist_ok=False)
    lock=(dest/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);start=time.monotonic()
    assert json.loads((O/'raw_rebuild_audit.json').read_text())['complete']
    fresh=np.load(O/'fresh_fm_cache_20261013/fresh_dual_cache.npz');oldcache=np.load(R/'fresh_fm_cache_20261013/fresh_dual_cache.npz')
    np.testing.assert_array_equal(fresh['issue_indices'],oldcache['issue_indices']);np.testing.assert_array_equal(fresh['group_present'],oldcache['group_present'])
    write(dest/'cache_comparison.json',dict(maximum_difference=float(np.max(np.abs(fresh['x']-oldcache['x']))),shape=list(fresh['x'].shape)))
    # Guard only regenerated identities; the archived module and source remain untouched.
    run.SOURCE_SHA=sha(O/'operational_qc_v3_regenerated_20261013/merged_qc_frame.csv.gz')
    run.CACHE_SHA=sha(O/'fresh_fm_cache_20261013/fresh_dual_cache.npz')
    run.SCALER_SHA=sha(O/BASE/'scalers.pt')
    d=run.data(O);tp=d['index'][d['train']]
    write(dest/'protocol.json',dict(seeds=[13,37,71],source_sha256=run.SOURCE_SHA,cache_sha256=run.CACHE_SHA,scaler_sha256=run.SCALER_SHA,architecture=ARCHS[2],variant=VARIANTS[0],train_indices_sha256=__import__('hashlib').sha256(d['train'].tobytes()).hexdigest(),base_epochs=50,relation_steps=640,post_relation_steps=40,repair_steps=80,plan_sft_steps=256,plan_grpo_steps=40,new_selection=False,originals_preserved=True,threads=4,memory_GiB=24,scope='selected model training lineage only, not all rejected development candidates'))
    metrics=[];comparisons=[]
    def gate():
        if time.monotonic()-start>21600:raise TimeoutError('6h gate: partial results preserved')
    def compare(stage,seed,state,prior):
        assert state.keys()==prior.keys()
        diff=max(float((state[k]-prior[k]).abs().max()) for k in state)
        comparisons.append(dict(stage=stage,seed=seed,max_parameter_difference=diff,bitexact=all(torch.equal(state[k],prior[k]) for k in state)))
        write(dest/'stage_comparison.json',comparisons)
    for seed in [13,37,71]:
        gate();sd=dest/f'seed{seed}';sd.mkdir();torch.manual_seed(seed);rng=np.random.default_rng(seed)
        legacy=Policy(288,216,d['fm'].shape[1]);cp=torch.load(O/BASE/f'checkpoint_seed{seed}.pt',weights_only=False)
        legacy.load_state_dict(cp['grpo'])
        oldbase=torch.load(R/BASE/f'checkpoint_seed{seed}.pt',weights_only=False)
        compare('base',seed,cp['grpo'],oldbase['grpo'])
        model=SixHourPolicy(legacy,ARCHS[2]);opt=torch.optim.Adam(model.parameters(),lr=3e-4)
        t=time.monotonic()
        for epoch in range(10):
            order=rng.permutation(tp)
            for left in range(0,len(order),64):
                at=order[left:left+64];loss=masked_loss(model(d['history'][at],d['fm'][at],d['plans'][0][at]),d['truth'][at])
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();assert torch.isfinite(loss)
                if epoch==0 and left==128:assert (time.monotonic()-t)/3*640<21600,'pilot budget gate'
            gate();print(json.dumps(dict(seed=seed,stage='relation_warm',epoch=epoch+1,seconds=time.monotonic()-t)),flush=True)
        warm=copy.deepcopy(model.state_dict());model.decoder_only();model.eval()
        e=run.encode(model,d)[:2];reference=copy.deepcopy(model)
        opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=3e-5);torch.manual_seed(seed+20000)
        for _ in range(20):
            at=rng.choice(tp,64,replace=False);grpo_update(model,reference,tuple(x[at] for x in e),d['plans'][0][at],d['truth'][at],opt)
        old=torch.load(R/'relational_ar_full_v2_20261016'/ARCHS[2]/f'seed{seed}/offline.pt',weights_only=False)
        compare('relation_warm',seed,warm,old['warm']);compare('relation_post',seed,model.state_dict(),old['post_grpo'])
        torch.save(dict(warm=warm,post_grpo=model.state_dict()),sd/'relational.pt')
        reference=copy.deepcopy(model);rng=np.random.default_rng(seed+42000)
        batches=[rng.choice(tp,64,replace=False) for _ in range(40)]
        torch.manual_seed(seed+43000);opt=CountAdam([p for p in model.parameters() if p.requires_grad],lr=3e-5)
        for at in batches:grpo_repair(model,reference,tuple(x[at] for x in e),d['plans'][0][at],d['truth'][at],opt,d['weights'][at],1.)
        assert opt.actual_steps==80
        old=torch.load(R/'online_ar_repair_20261016'/f'seed{seed}/총량저속보상.pt',weights_only=False)
        compare('repair',seed,model.state_dict(),old['state']);torch.save(model.state_dict(),sd/'repair.pt')
        e=run.encode(model,d);torch.manual_seed(seed+19000);active=PlanResponsePolicy(model,VARIANTS[0]);rng=np.random.default_rng(seed+19100)
        batches=[rng.choice(tp,64,replace=False) for _ in range(276)]
        opt=CountAdam([p for p in active.parameters() if p.requires_grad],lr=3e-4)
        for at in batches[:256]:
            loss=learning_loss(active,tuple(x[at] for x in e),d['plans'][0][at],d['truth'][at],d['weights'][at]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in active.parameters() if p.requires_grad],1.);opt.step();assert torch.isfinite(loss)
        assert opt.actual_steps==256;reference=copy.deepcopy(active);torch.manual_seed(seed+19200)
        opt=CountAdam([p for p in active.parameters() if p.requires_grad],lr=3e-5)
        for at in batches[256:]:conditioned_grpo(active,reference,tuple(x[at] for x in e),d['plans'][0][at],d['truth'][at],opt,d['weights'][at])
        assert opt.actual_steps==40 and all(torch.isfinite(p).all() for p in active.parameters())
        name='인과계획학습__동일예산기존형__GRPO고정'
        old=torch.load(R/'plan_response_full_v2_20261019'/f'seed{seed}'/(name+'.pt'),weights_only=False)
        compare('final',seed,active.state_dict(),old['state']);torch.save(dict(state=active.state_dict()),sd/'final.pt')
        active.eval()
        for split in ['validation','test']:
            ids=d['splits'][split];pred=run.predict(active,e,d['plans'][0],d['index'][ids]);truth=d['target'][ids]
            prior=np.load(R/'plan_response_full_v2_20261019'/f'seed{seed}'/(split+'__'+name+'__발행가용모의계획.npz'))
            np.testing.assert_array_equal(ids,prior['issues']);np.testing.assert_allclose(truth,prior['target'],equal_nan=True,atol=0,rtol=0)
            np.savez_compressed(sd/(split+'.npz'),prediction=pred,target=truth,issues=ids)
            full=np.isfinite(truth).all(1)
            for segment,y,p in [('all',truth.ravel(),pred.ravel()),('first',truth[:,0],pred[:,0]),('total',truth[full].sum(1),pred[full].sum(1))]:
                metrics.append(dict(seed=seed,split=split,segment=segment,max_prediction_difference=float(np.max(np.abs(pred-prior['prediction']))),**score(y,p)))
        pd.DataFrame(metrics).to_csv(dest/'metrics.csv',index=False);print('SEED_COMPLETE',seed,flush=True)
    write(dest/'completion.json',dict(complete=True,seconds=time.monotonic()-start,seeds=[13,37,71],new_selection=False));print('FULL_SELECTED_CHAIN_COMPLETE',flush=True)
    subprocess.run([sys.executable,str(R/'submission_baseline_rerun_20261026.py')],check=True,timeout=max(1,21600-(time.monotonic()-start)))
    write(O/'completion.json',dict(complete=True,selected_training=True,baseline_reexecution=True,raw_path_verified=True))
if __name__=='__main__':main()
