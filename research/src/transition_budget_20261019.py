"""Forecasted auxiliary operational state; actual future states never enter forward."""
import copy
import torch
from torch import nn
import torch.nn.functional as F
from plan_response_20261019 import plan_features
from idle_budget_20261019 import inverse_softplus,complete_loss
from online_ar_repair_20261016 import row_cost

VARIANTS=['동일예산전환미사용','운항전환잔차','운항전환총량결합']
class TransitionPolicy(nn.Module):
    def __init__(self,parent,variant,low_rate):
        super().__init__();self.parent=copy.deepcopy(parent);self.variant=variant;self.parent.decoder_only();self.register_buffer('low_rate',torch.tensor(float(low_rate)))
        self.aux=nn.Sequential(nn.Linear(105,32),nn.Tanh(),nn.Linear(32,1))
        if variant!=VARIANTS[0]:
            self.residual=nn.Sequential(nn.Linear(26,16),nn.Tanh(),nn.Linear(16,1));nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
    def auxiliary(self,e,plans):
        ctx,_,raw,source,recent=e;features=plan_features(raw,plans)
        return self.aux(torch.cat([ctx[:,None].expand(-1,12,-1),features,recent[:,None].expand(-1,12,-1)],-1)).squeeze(-1)
    def freeze_auxiliary(self):
        for p in self.aux.parameters():p.requires_grad_(False);p.grad=None
    def conditional(self,e,plans,actions=None,noise=None):
        ctx,memory,raw,source,recent=e;par=self.parent;core=par.backbone;base=core.head(ctx[:,None]+torch.tanh(core.plan(plans)))[:,:,1];features=plan_features(raw,plans)
        delta=par.source_bias*source.expand(-1,12)+2*torch.tanh(par.response(torch.cat([features,source[:,None].expand(-1,12,-1)],-1)).squeeze(-1))
        if hasattr(self,'residual'):
            predicted_score=torch.sigmoid(self.auxiliary(e,plans)).detach();gate=predicted_score*torch.tanh(self.residual(torch.cat([features,source[:,None].expand(-1,12,-1),recent[:,None].expand(-1,12,-1)],-1)).squeeze(-1))
            current=torch.exp(-(raw[:,0]*50/3).square())*raw[:,9]*recent[:,1];ref=inverse_softplus(current*recent[:,0]+(1-current)*self.low_rate)
        projected=core.cross.project(memory);state=torch.tanh(core.initial(ctx));previous=F.softplus(base[:,0]).detach();values=[];chosen=[]
        for k in range(12):
            attended=core.cross(state[:,None],projected);state=core.gru(torch.cat([previous[:,None],plans[:,k],attended[:,0]],1),state);value=base[:,k]+core.readout(state).squeeze(-1)+delta[:,k]
            if hasattr(self,'residual'):value=value+gate[:,k]*(ref-value)
            action=actions[:,k] if actions is not None else value+(noise[:,k] if noise is not None else 0);values.append(value);chosen.append(action);previous=F.softplus(action)
        return torch.stack(values,1),torch.stack(chosen,1)
    def predict_encoded(self,e,plans):return F.softplus(self.conditional(e,plans)[0])

def loss(model,e,plans,target,weights,aux_target=None):
    pred=model.predict_encoded(e,plans)
    if model.variant==VARIANTS[-1]:objective=complete_loss(pred,target,weights)
    else:
        objective=(row_cost(pred,target,1,True)*weights).sum()/weights.sum();valid=torch.isfinite(target[:,:-1])&torch.isfinite(target[:,1:])
        if valid.any():
            delta=torch.diff(pred,dim=1)-torch.diff(torch.nan_to_num(target),dim=1);objective=objective+.25*F.smooth_l1_loss(delta[valid],torch.zeros_like(delta[valid]),beta=.2)
    if aux_target is not None:
        known=torch.isfinite(aux_target)
        if known.any():objective=objective+.5*F.binary_cross_entropy_with_logits(model.auxiliary(e,plans)[known],aux_target[known],pos_weight=torch.tensor(4.))
    return objective

def update(model,reference,e,plans,target,opt,weights):
    from idle_budget_20261019 import reward_cost
    from idle_budget_20261019 import VARIANTS as OLD
    b=len(plans);std=.15;ext=tuple(x.repeat(8,*([1]*(x.ndim-1))) for x in e);ep=plans.repeat(8,1,1);valid=torch.isfinite(target).repeat(8,1)
    with torch.no_grad():
        old,actions=model.conditional(ext,ep,noise=torch.randn(8*b,12)*std);cost=reward_cost(F.softplus(actions),target.repeat(8,1),OLD[1] if model.variant==VARIANTS[-1] else OLD[0]);reward=-cost.reshape(8,b);adv=(reward-reward.mean(0))/reward.std(0,unbiased=False).clamp_min(1e-6);oldlp=(torch.distributions.Normal(old,std).log_prob(actions)*valid).sum(1);ref=reference.conditional(ext,ep,actions=actions)[0]
    for _ in range(2):
        cur=model.conditional(ext,ep,actions=actions)[0];lp=(torch.distributions.Normal(cur,std).log_prob(actions)*valid).sum(1);ratio=(lp-oldlp).clamp(-20,20).exp().reshape(8,b);pg=(torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)*weights[None]).sum()/(8*weights.sum());kl=((cur-ref).square()[valid]/(2*std**2)).mean();objective=-pg+.05*kl+.5*loss(model,e,plans,target,weights)
        opt.zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);opt.step()
