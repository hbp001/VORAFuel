"""Six-hour variable-time relational encoder and one conditional fuel policy.

Frozen Chronos cache, retained legacy GRPO initialization, no ensembles.
Inference never accepts target fuel. Units inside the head: MT/30min divided by .3.
"""
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from operational_core import VESSEL, WEATHER, causal_channels
from planned_multiscale_20261015 import endpoint_labels

ARCHS=['기존FM확장직접형','관계학습직접형','관계학습AR형']
SCALE=.3


def six_hour_targets(frame):
    n=len(frame);_,half,_=endpoint_labels(frame)
    y=np.full((n,12),np.nan,np.float32)
    ends=np.arange(n)[:,None]+np.arange(3,37,3)[None,:]
    for j,h in enumerate(range(3,37,3)):y[:-h,j]=half[h:]
    return y,ends+3


def issue_split(frame,split):
    start,end={'train':('2025-01-01','2026-09-11'),'validation':('2026-09-11','2026-09-21'),'test':('2026-09-21','2026-10-01')}[split]
    ids=np.flatnonzero((frame.index>=start)&(frame.index<end)&(frame.index.minute%30==0))
    return ids[(ids>=147)&(ids+36<np.searchsorted(frame.index,pd.Timestamp(end)))&(ids+36<len(frame))]


def histories(frame,scalers):
    """36 native 10min samples: last received t-30min through t-380min."""
    blocks=[]
    for (values,mask,age),s in zip(causal_channels(frame,3),scalers['channel_scalers']):
        z=np.nan_to_num((values-np.asarray(s['median']))/np.asarray(s['std'])).clip(-10,10)
        base=np.stack([z,mask,age],axis=-1)
        window=np.stack([np.roll(base,lag,axis=0) for lag in range(36)],axis=1)
        for lag in range(36):window[:lag,lag]=0
        blocks.append(window.astype(np.float32))
    return np.concatenate(blocks,axis=2)


def expand_projection(layer,variables):
    new=nn.Linear(36*variables*3,layer.out_features)
    with torch.no_grad():
        new.weight.zero_();new.weight[:,:12*variables*3].copy_(layer.weight);new.bias.copy_(layer.bias)
    return new


class RelationEncoder(nn.Module):
    """Keep variable identity and lag; alternating time and variable attention."""
    def __init__(self,d=24,variables=14):
        super().__init__();self.variables=variables;self.d=d
        self.vessel_in=nn.Linear(9,d);self.environment_in=nn.Linear(9,d)
        self.variable=nn.Embedding(variables,d);self.lag=nn.Embedding(12,d)
        self.time_layer=nn.TransformerEncoderLayer(d,4,d*2,dropout=0,batch_first=True,norm_first=True)
        self.variable_layer=nn.TransformerEncoderLayer(d,4,d*2,dropout=0,batch_first=True,norm_first=True)
        self.norm=nn.LayerNorm(d)

    def forward(self,history):
        b=history.shape[0]
        # Each patch retains all three native values/masks/ages, not a mean fill.
        patches=history.reshape(b,12,3,self.variables,3).permute(0,1,3,2,4).reshape(b,12,self.variables,9)
        z=torch.cat([self.vessel_in(patches[:,:,:8]),self.environment_in(patches[:,:,8:])],dim=2)
        z=z+self.variable.weight[None,None]+self.lag.weight[None,:,None]
        z=self.time_layer(z.permute(0,2,1,3).reshape(b*self.variables,12,self.d)).reshape(b,self.variables,12,self.d).permute(0,2,1,3)
        z=self.variable_layer(z.reshape(b*12,self.variables,self.d)).reshape(b,12*self.variables,self.d)
        return self.norm(z)


class ReadAttention(nn.Module):
    """Project immutable memory once per rollout, not once per AR step."""
    def __init__(self,d=24,heads=4):
        super().__init__();self.heads=heads;self.d=d
        self.q=nn.Linear(d,d);self.k=nn.Linear(d,d);self.v=nn.Linear(d,d);self.out=nn.Linear(d,d)

    def project(self,memory):
        b,n,_=memory.shape
        return tuple(layer(memory).reshape(b,n,self.heads,self.d//self.heads).transpose(1,2) for layer in [self.k,self.v])

    def forward(self,query,projected):
        b,n,_=query.shape;q=self.q(query).reshape(b,n,self.heads,self.d//self.heads).transpose(1,2)
        result=F.scaled_dot_product_attention(q,*projected,dropout_p=0)
        return self.out(result.transpose(1,2).reshape(b,n,self.d))


class SixHourPolicy(nn.Module):
    def __init__(self,legacy,architecture):
        super().__init__()
        if architecture not in ARCHS:raise ValueError(architecture)
        self.architecture=architecture
        self.vessel=copy.deepcopy(legacy.vessel);self.weather=copy.deepcopy(legacy.weather);self.fm=copy.deepcopy(legacy.fm)
        self.vessel[0]=expand_projection(legacy.vessel[0],len(VESSEL))
        self.weather[0]=expand_projection(legacy.weather[0],len(WEATHER))
        self.head=copy.deepcopy(legacy.head)
        self.plan=nn.Linear(9,80);nn.init.zeros_(self.plan.weight);nn.init.zeros_(self.plan.bias)
        if architecture!=ARCHS[0]:
            self.relation=RelationEncoder()
            self.query=nn.Linear(89,24)
            self.cross=ReadAttention()
            self.readout=nn.Linear(24,1);nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)
            if architecture==ARCHS[2]:
                self.initial=nn.Linear(80,24)
                self.gru=nn.GRUCell(34,24)

    def encode(self,history,fm):
        b=history.shape[0]
        # Same order as the legacy z/mask/age flattened window, newest first.
        def flat(h):return h.permute(0,1,3,2).reshape(b,-1)
        ctx=torch.cat([self.vessel(flat(history[:,:,:8])),self.weather(flat(history[:,:,8:])),self.fm(fm)],dim=1)
        memory=self.relation(history) if self.architecture!=ARCHS[0] else ctx[:,None,:24]*0
        return ctx,memory

    def conditional(self,encoded,plans,actions=None,noise=None):
        """Actions/noise are predicted logit actions, NEVER ground-truth targets."""
        ctx,memory=encoded;b=ctx.shape[0]
        base=self.head(ctx[:,None]+torch.tanh(self.plan(plans)))[:,:,1]
        if self.architecture==ARCHS[0]:
            chosen=base if noise is None else base+noise
            return base,chosen
        projected=self.cross.project(memory)
        if self.architecture==ARCHS[1]:
            q=self.query(torch.cat([ctx[:,None].expand(-1,12,-1),plans],dim=-1))
            attended=self.cross(q,projected)
            logits=base+self.readout(torch.tanh(attended)).squeeze(-1)
            return logits,logits if noise is None else logits+noise
        state=torch.tanh(self.initial(ctx));previous=F.softplus(base[:,0]).detach()
        logits=[];chosen=[]
        for k in range(12):
            attended=self.cross(state[:,None],projected)
            state=self.gru(torch.cat([previous[:,None],plans[:,k],attended[:,0]],dim=1),state)
            value=base[:,k]+self.readout(state).squeeze(-1)
            action=actions[:,k] if actions is not None else value+(noise[:,k] if noise is not None else 0)
            logits.append(value);chosen.append(action);previous=F.softplus(action)
        return torch.stack(logits,1),torch.stack(chosen,1)

    def predict_encoded(self,encoded,plans):return F.softplus(self.conditional(encoded,plans)[0])
    def forward(self,history,fm,plans):return self.predict_encoded(self.encode(history,fm),plans)

    def decoder_only(self):
        for name,param in self.named_parameters():
            param.requires_grad_(not name.startswith(('vessel.','weather.','fm.','relation.')))
            # Freezing does not clear a previous backward's gradients. Leaving
            # them attached would corrupt the decoder-only clipping norm.
            param.grad=None


def masked_loss(pred,target):
    valid=torch.isfinite(target)
    if not valid.any():raise ValueError('No released targets')
    return F.smooth_l1_loss(pred[valid],target[valid],beta=.2)+.15*(pred[valid]-target[valid]).square().mean()


def released_values(values,release,issue):
    """Never expose an unreleased target even when stored for later evaluation."""
    result=np.asarray(values,float).copy()
    result[(np.asarray(release)>issue)|~np.isfinite(result)]=np.nan
    return result


def grpo_update(model,reference,encoded,plans,target,opt):
    """Eight positive autoregressive action trajectories; mask unreleased truth."""
    valid=torch.isfinite(target);b=len(plans);std=.15
    if not valid.any():raise ValueError('No released targets')
    ext=tuple(x.repeat(8,*([1]*(x.ndim-1))) for x in encoded)
    ep=plans.repeat(8,1,1);vm=valid.repeat(8,1)
    with torch.no_grad():
        old,actions=model.conditional(ext,ep,noise=torch.randn(8*b,12,device=plans.device)*std)
        values=F.softplus(actions).reshape(8,b,12)
        error=torch.where(valid[None],values-target[None],0)
        reward=-(error.abs()+.15*error.square()).sum(-1)/valid.sum(-1).clamp_min(1)
        advantage=(reward-reward.mean(0))/reward.std(0,unbiased=False).clamp_min(1e-6)
        old_lp=(torch.distributions.Normal(old,std).log_prob(actions)*vm).sum(1)
        ref=reference.conditional(ext,ep,actions=actions)[0]
    for _ in range(2):
        current=model.conditional(ext,ep,actions=actions)[0]
        lp=(torch.distributions.Normal(current,std).log_prob(actions)*vm).sum(1)
        ratio=torch.exp((lp-old_lp).clamp(-20,20)).reshape(8,b)
        pg=torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
        kl=((current-ref).square()[vm]/(2*std**2)).mean()
        loss=-pg+.05*kl+.5*masked_loss(model.predict_encoded(encoded,plans),target)
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step()
