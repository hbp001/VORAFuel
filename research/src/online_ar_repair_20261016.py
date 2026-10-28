"""Single-policy, released-label-only aggregate/state GRPO and bounded updates."""
import copy
import numpy as np
import torch
import torch.nn.functional as F

CALIBRATIONS=['미보완점보상','총량보상','총량저속보상','총량저속계획완화','동일목적SFT']

def state_weights(speed,observed):
    speed=np.asarray(speed);observed=np.asarray(observed,bool)
    return np.where(observed&np.isfinite(speed)&(speed<=3),4.,1.).astype(np.float32)

def drop_plan_information(plan,drop):
    """Remove optional operational plan, never alter lead/calendar tokens."""
    result=plan.clone();result[drop,: ,:5]=0
    return result

def row_cost(pred,target,total_weight=0.,huber=False):
    valid=torch.isfinite(target)
    if not valid.any(1).all():raise ValueError('Every sampled trajectory needs a released target')
    safe=torch.where(valid,target,torch.zeros_like(target))
    err=torch.where(valid,pred-safe,torch.zeros_like(pred))
    point=(F.smooth_l1_loss(err,torch.zeros_like(err),beta=.2,reduction='none') if huber else err.abs())+.15*err.square()
    cost=(point*valid).sum(1)/valid.sum(1)
    # Complete observed 6h trajectories only. Dividing the total by 12 keeps units/balance.
    complete=valid.all(1);bias=(pred-safe).mean(1)
    aggregate=(F.smooth_l1_loss(bias,torch.zeros_like(bias),beta=.2,reduction='none') if huber else bias.abs())+.15*bias.square()
    return cost+total_weight*torch.where(complete,aggregate,torch.zeros_like(aggregate))

def objective(pred,target,weights,total_weight):
    return (row_cost(pred,target,total_weight,huber=True)*weights).sum()/weights.sum()

def grpo_repair(model,reference,encoded,plans,target,opt,weights,total_weight=0.,kl_weight=.05,prediction_anchor=0.,steps=2):
    valid=torch.isfinite(target);b=len(plans);std=.15
    if not valid.any(1).all():raise ValueError('Unreleased/empty trajectories cannot be sampled')
    ext=tuple(x.repeat(8,*([1]*(x.ndim-1))) for x in encoded);ep=plans.repeat(8,1,1);vm=valid.repeat(8,1)
    with torch.no_grad():
        old,actions=model.conditional(ext,ep,noise=torch.randn(8*b,12,device=plans.device)*std)
        values=F.softplus(actions)
        reward=-row_cost(values,target.repeat(8,1),total_weight).reshape(8,b)
        adv=(reward-reward.mean(0))/reward.std(0,unbiased=False).clamp_min(1e-6)
        old_lp=(torch.distributions.Normal(old,std).log_prob(actions)*vm).sum(1)
        ref=reference.conditional(ext,ep,actions=actions)[0]
        refpred=reference.predict_encoded(encoded,plans)
    for _ in range(steps):
        current=model.conditional(ext,ep,actions=actions)[0]
        lp=(torch.distributions.Normal(current,std).log_prob(actions)*vm).sum(1)
        ratio=torch.exp((lp-old_lp).clamp(-20,20)).reshape(8,b)
        pg=(torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)*weights[None]).sum()/(8*weights.sum())
        kl=((current-ref).square()[vm]/(2*std**2)).mean()
        pred=model.predict_encoded(encoded,plans)
        loss=-pg+kl_weight*kl+.5*objective(pred,target,weights,total_weight)
        if prediction_anchor:loss=loss+prediction_anchor*F.smooth_l1_loss(pred,refpred,beta=.2)
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        opt.step()

def protected_decision(initial,after,before_dynamic,after_dynamic):
    """Absolute anchor prevents repeated per-step tolerances from compounding."""
    for key,value in initial.items():
        if np.isfinite(value):
            if not np.isfinite(after[key]) or after[key]>value*1.01+1e-8:return False,'초기기준_'+key
    if np.isfinite(before_dynamic) and (not np.isfinite(after_dynamic) or after_dynamic>before_dynamic+1e-8):return False,'수신정답보류검사'
    return True,'승인'

def rollback(model,opt,model_state,opt_state):
    model.load_state_dict(model_state);opt.load_state_dict(opt_state)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in model_state.items())

def choose_validation(table,baseline):
    """Select exactly once on validation; test data are not accepted here."""
    choices=[]
    for name in CALIBRATIONS[:4]:
        m=table[name]
        eligible=m['point']<=baseline['point']*1.03 and m['total']<=baseline['total'] and m['slow']<=baseline['slow']
        if eligible:choices.append((m['point']/baseline['point']+.5*m['total']/baseline['total']+.5*m['slow']/baseline['slow'],CALIBRATIONS.index(name),name))
    return min(choices)[2] if choices else None
