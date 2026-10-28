"""One AR policy with explicit plan relationships and nonzero low-load residual."""
import copy
import torch
from torch import nn
import torch.nn.functional as F
from online_ar_repair_20261016 import objective

VARIANTS=['동일예산기존형','관계조건특징형','관계조건준정지형','조건반응강화형']

def plan_features(raw,plans):
    """raw: received SOG/50, fuel*10, heading sin/cos, wind uv/10,
    wave/3, wave sin/cos, fresh SOG/fuel/heading/wind/wave masks."""
    b,n,_=plans.shape;r=raw[:,None].expand(-1,n,-1);valid=plans[:,:,3];hdvalid=plans[:,:,4]
    v=plans[:,:,0]*valid;s=plans[:,:,1]*hdvalid;c=plans[:,:,2]*hdvalid
    previous_s=torch.cat([raw[:,None,2],s[:,:-1]],1);previous_c=torch.cat([raw[:,None,3],c[:,:-1]],1)
    previous_valid=torch.cat([raw[:,None,11],hdvalid[:,:-1]],1)
    turn=(1-(s*previous_s+c*previous_c).clamp(-1,1))*hdvalid*previous_valid
    previous_v=torch.cat([raw[:,None,0],v[:,:-1]],1)
    z=[v,v-r[:,:,0],v.square(),v.pow(3),r[:,:,0],r[:,:,1],valid,hdvalid,
       (s*r[:,:,2]+c*r[:,:,3])*r[:,:,11]*hdvalid,(s*r[:,:,3]-c*r[:,:,2])*r[:,:,11]*hdvalid,
       (s*r[:,:,4]+c*r[:,:,5])*r[:,:,12]*hdvalid,(c*r[:,:,4]-s*r[:,:,5])*r[:,:,12]*hdvalid,
       r[:,:,6]*(s*r[:,:,7]+c*r[:,:,8])*r[:,:,13]*hdvalid,
       r[:,:,6]*(s*r[:,:,8]-c*r[:,:,7])*r[:,:,13]*hdvalid,
       turn,(v-previous_v)*valid,r[:,:,9],r[:,:,10],r[:,:,11],plans[:,:,5]*4,plans[:,:,6],plans[:,:,7]]
    return torch.stack(z,-1)

class PlanResponsePolicy(nn.Module):
    def __init__(self,backbone,variant):
        super().__init__();self.backbone=copy.deepcopy(backbone);self.variant=variant
        self.source_bias=nn.Parameter(torch.zeros(()))
        if variant!=VARIANTS[0]:
            self.response=nn.Sequential(nn.Linear(23,24),nn.Tanh(),nn.Linear(24,1))
            nn.init.zeros_(self.response[-1].weight);nn.init.zeros_(self.response[-1].bias)
        if variant in VARIANTS[2:]:
            self.low=nn.Sequential(nn.Linear(23,12),nn.Tanh(),nn.Linear(12,1))
            nn.init.zeros_(self.low[-1].weight);nn.init.zeros_(self.low[-1].bias)
        self.decoder_only()

    def decoder_only(self):
        self.backbone.decoder_only()
        for p in self.parameters():p.grad=None

    def conditional(self,encoded,plans,actions=None,noise=None):
        ctx,memory,raw,source=encoded;core=self.backbone;b=len(ctx)
        base=core.head(ctx[:,None]+torch.tanh(core.plan(plans)))[:,:,1]
        delta=self.source_bias*source.expand(-1,12)
        if self.variant!=VARIANTS[0]:
            features=torch.cat([plan_features(raw,plans),source[:,None].expand(-1,12,-1)],-1)
            delta=delta+2*torch.tanh(self.response(features).squeeze(-1))
            if self.variant in VARIANTS[2:]:
                current=torch.exp(-(raw[:,0]*50/3).square())*raw[:,9]
                planned=torch.exp(-(plans[:,:,0]*50/3).square())*plans[:,:,3]
                gate=(1-source)*current[:,None]+source*planned
                delta=delta+gate*3*torch.tanh(self.low(features).squeeze(-1))
        projected=core.cross.project(memory);state=torch.tanh(core.initial(ctx));previous=F.softplus(base[:,0]).detach();values=[];chosen=[]
        for k in range(12):
            attended=core.cross(state[:,None],projected)
            state=core.gru(torch.cat([previous[:,None],plans[:,k],attended[:,0]],1),state)
            value=base[:,k]+core.readout(state).squeeze(-1)+delta[:,k]
            action=actions[:,k] if actions is not None else value+(noise[:,k] if noise is not None else 0)
            values.append(value);chosen.append(action);previous=F.softplus(action)
        return torch.stack(values,1),torch.stack(chosen,1)

    def predict_encoded(self,encoded,plans):return F.softplus(self.conditional(encoded,plans)[0])

def speed_scenarios(plans):
    """Change supplied speed slots only; preserve unknown AR prefix slots."""
    up=plans.clone();down=plans.clone();known=plans[:,:,3]>0
    up[:,:,0]=torch.where(known,(plans[:,:,0]+.06).clamp(0,1),plans[:,:,0])
    down[:,:,0]=torch.where(known,(plans[:,:,0]-.06).clamp(0,1),plans[:,:,0])
    return up,down

def learning_loss(model,encoded,plans,target,weights):
    pred=model.predict_encoded(encoded,plans);loss=objective(pred,target,weights,1)
    adjacent=torch.isfinite(target[:,:-1])&torch.isfinite(target[:,1:])
    if adjacent.any():
        delta=torch.diff(pred,dim=1)-torch.diff(torch.nan_to_num(target),dim=1)
        loss=loss+.25*F.smooth_l1_loss(delta[adjacent],torch.zeros_like(delta[adjacent]),beta=.2)
    if model.variant==VARIANTS[3]:
        up,down=speed_scenarios(plans)
        eligible=(plans[:,:,3]>0)&(plans[:,:,0]>.06)&(plans[:,:,0]<.94)
        if eligible.any():
            diff=model.predict_encoded(encoded,up)-model.predict_encoded(encoded,down)
            confidence=.25+.75*encoded[3]
            loss=loss+.5*(F.relu(.02-diff)*confidence)[eligible].mean()
    return loss

def conditioned_grpo(model,reference,e,plans,target,opt,weights):
    """Same group8, two updates; complete totals only, no target in prefix."""
    from online_ar_repair_20261016 import row_cost
    valid=torch.isfinite(target);b=len(plans);std=.15
    ext=tuple(x.repeat(8,*([1]*(x.ndim-1))) for x in e);ep=plans.repeat(8,1,1);vm=valid.repeat(8,1)
    with torch.no_grad():
        old,actions=model.conditional(ext,ep,noise=torch.randn(8*b,12)*std)
        reward=-row_cost(F.softplus(actions),target.repeat(8,1),1).reshape(8,b)
        advantage=(reward-reward.mean(0))/reward.std(0,unbiased=False).clamp_min(1e-6)
        old_lp=(torch.distributions.Normal(old,std).log_prob(actions)*vm).sum(1)
        ref=reference.conditional(ext,ep,actions=actions)[0]
    for _ in range(2):
        current=model.conditional(ext,ep,actions=actions)[0]
        lp=(torch.distributions.Normal(current,std).log_prob(actions)*vm).sum(1)
        ratio=torch.exp((lp-old_lp).clamp(-20,20)).reshape(8,b)
        pg=(torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage)*weights[None]).sum()/(8*weights.sum())
        kl=((current-ref).square()[vm]/(2*std**2)).mean()
        loss=-pg+.05*kl+.5*learning_loss(model,e,plans,target,weights)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step()
