"""One AR policy: observed-rate innovation and complete-trajectory budget loss."""
import copy
import torch
from torch import nn
import torch.nn.functional as F
from plan_response_20261019 import plan_features,learning_loss
from online_ar_repair_20261016 import row_cost

VARIANTS=['동일예산연장','완전총량분리정규화','최근연료잔차','최근연료총량결합','최근연료공유총량잔차']

def inverse_softplus(x):
    x=x.clamp_min(1e-4)
    return x+torch.log(-torch.expm1(-x))

class IdleBudgetPolicy(nn.Module):
    def __init__(self,parent,variant):
        super().__init__();self.parent=copy.deepcopy(parent);self.variant=variant;self.decoder_only()
        if variant in VARIANTS[2:]:
            self.anchor=nn.Sequential(nn.Linear(26,16),nn.Tanh(),nn.Linear(16,1));nn.init.zeros_(self.anchor[-1].weight);nn.init.zeros_(self.anchor[-1].bias)
        if variant==VARIANTS[4]:
            self.budget=nn.Sequential(nn.Linear(105,16),nn.Tanh(),nn.Linear(16,1));nn.init.zeros_(self.budget[-1].weight);nn.init.zeros_(self.budget[-1].bias)
    def decoder_only(self):
        self.parent.decoder_only()
        for p in self.parameters():p.grad=None
    def conditional(self,e,plans,actions=None,noise=None):
        ctx,memory,raw,source,recent=e;par=self.parent;core=par.backbone
        base=core.head(ctx[:,None]+torch.tanh(core.plan(plans)))[:,:,1]
        features=plan_features(raw,plans);delta=par.source_bias*source.expand(-1,12)+2*torch.tanh(par.response(torch.cat([features,source[:,None].expand(-1,12,-1)],-1)).squeeze(-1))
        shift=2*torch.tanh(self.budget(torch.cat([ctx,features.mean(1),recent],1))) if hasattr(self,'budget') else torch.zeros_like(source)
        if hasattr(self,'anchor'):
            a=torch.tanh(self.anchor(torch.cat([features,source[:,None].expand(-1,12,-1),recent[:,None].expand(-1,12,-1)],-1)).squeeze(-1))
            gate=torch.exp(-(raw[:,0]*50/3).square())*raw[:,9]*recent[:,1];a=a*gate[:,None]
            reference=inverse_softplus(recent[:,0])
        projected=core.cross.project(memory);state=torch.tanh(core.initial(ctx));previous=F.softplus(base[:,0]).detach();values=[];chosen=[]
        for k in range(12):
            attended=core.cross(state[:,None],projected);state=core.gru(torch.cat([previous[:,None],plans[:,k],attended[:,0]],1),state)
            value=base[:,k]+core.readout(state).squeeze(-1)+delta[:,k]+shift[:,0]
            if hasattr(self,'anchor'):value=value+a[:,k]*(reference-value)
            action=actions[:,k] if actions is not None else value+(noise[:,k] if noise is not None else 0)
            values.append(value);chosen.append(action);previous=F.softplus(action)
        return torch.stack(values,1),torch.stack(chosen,1)
    def predict_encoded(self,e,plans):return F.softplus(self.conditional(e,plans)[0])

def enhanced(variant):return variant in [VARIANTS[1],VARIANTS[3],VARIANTS[4]]

def aggregate_term(pred,target,width,huber):
    valid=torch.isfinite(target[:,:width]).all(1);safe=torch.nan_to_num(target[:,:width]);bias=(pred[:,:width]-safe).mean(1)
    value=(F.smooth_l1_loss(bias,torch.zeros_like(bias),beta=.2,reduction='none') if huber else bias.abs())+.15*bias.square()
    return value,valid

def complete_loss(pred,target,weights):
    point=(row_cost(pred,target,0,True)*weights).sum()/weights.sum()
    for width,coefficient in [(12,2.),(2,.25),(6,.25)]:
        val,valid=aggregate_term(pred,target,width,True)
        if valid.any():point=point+coefficient*(val[valid]*weights[valid]).sum()/weights[valid].sum()
    adjacent=torch.isfinite(target[:,:-1])&torch.isfinite(target[:,1:])
    if adjacent.any():
        delta=torch.diff(pred,dim=1)-torch.diff(torch.nan_to_num(target),dim=1)
        point=point+.25*F.smooth_l1_loss(delta[adjacent],torch.zeros_like(delta[adjacent]),beta=.2)
    return point

def training_loss(model,e,plans,target,weights):
    if not enhanced(model.variant):
        # Exactly the prior feature-model objective, without speed-rank prior.
        pred=model.predict_encoded(e,plans);loss=(row_cost(pred,target,1,True)*weights).sum()/weights.sum()
        valid=torch.isfinite(target[:,:-1])&torch.isfinite(target[:,1:])
        if valid.any():loss=loss+.25*F.smooth_l1_loss((torch.diff(pred,dim=1)-torch.diff(torch.nan_to_num(target),dim=1))[valid],torch.zeros_like(pred[:,:-1][valid]),beta=.2)
        return loss
    return complete_loss(model.predict_encoded(e,plans),target,weights)

def reward_cost(pred,target,variant):
    if not enhanced(variant):return row_cost(pred,target,1)
    cost=row_cost(pred,target,0)
    for width,coefficient in [(12,2.),(2,.25),(6,.25)]:
        value,valid=aggregate_term(pred,target,width,False)
        cost=cost+coefficient*torch.where(valid,value,torch.zeros_like(value))/valid.float().mean().clamp_min(1e-6)
    return cost

def grpo(model,reference,e,plans,target,opt,weights):
    b=len(plans);std=.15;valid=torch.isfinite(target).repeat(8,1);ext=tuple(x.repeat(8,*([1]*(x.ndim-1))) for x in e);ep=plans.repeat(8,1,1)
    with torch.no_grad():
        old,actions=model.conditional(ext,ep,noise=torch.randn(8*b,12)*std);reward=-reward_cost(F.softplus(actions),target.repeat(8,1),model.variant).reshape(8,b)
        adv=(reward-reward.mean(0))/reward.std(0,unbiased=False).clamp_min(1e-6);oldlp=(torch.distributions.Normal(old,std).log_prob(actions)*valid).sum(1);ref=reference.conditional(ext,ep,actions=actions)[0]
    for _ in range(2):
        current=model.conditional(ext,ep,actions=actions)[0];lp=(torch.distributions.Normal(current,std).log_prob(actions)*valid).sum(1);ratio=(lp-oldlp).clamp(-20,20).exp().reshape(8,b)
        pg=(torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)*weights[None]).sum()/(8*weights.sum());kl=((current-ref).square()[valid]/(2*std**2)).mean()
        loss=-pg+.05*kl+.5*training_loss(model,e,plans,target,weights);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);opt.step()
