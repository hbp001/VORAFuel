"""Causal plan-conditioned single fuel policy; 10min + 48 half-hour bins.

All targets are rates in MT/10min internally; long bins multiply by three for
MT/30min. A synthetic past-day plan is NOT an actual filed voyage plan.
"""
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn

from operational_core import Policy, raw_target
from short_gap_labels import NATIVE

MODES = ['계획없음', '전일항적모의계획', '실제미래운항oracle']
ARMS = ['가중치고정', '온라인SFT', '온라인GRPO']
SCALE = .1


def endpoint_labels(frame):
    """Thirty-minute observed cumulative differences; never impute a counter."""
    total = frame.cam2_total.where(frame.cam2_total.between(10000, 100000))
    parts = (frame.cam2_hfo+frame.cam2_mdo).where((frame.cam2_hfo+frame.cam2_mdo).between(10000, 100000))
    level = pd.concat([total, parts], axis=1).median(axis=1, skipna=True)
    agree = total.notna() & parts.notna() & ((total-parts).abs() <= .02)
    present = frame[['cam2_total','cam2_hfo','cam2_mdo']].notna().any(axis=1)
    sensor = frame[[c for c in NATIVE if c in frame]].notna().any(axis=1)
    rejected = present & ~agree
    # An observed counter drop anywhere within the interval invalidates it.
    previous_valid = level.where(agree).ffill(limit=3)
    reset = agree & (level < previous_valid.shift(1))
    ok = agree & agree.shift(3, fill_value=False)
    for lag in range(3):
        ok &= sensor.shift(lag, fill_value=False) & ~rejected.shift(lag, fill_value=True) & ~reset.shift(lag, fill_value=True)
    delta = level-level.shift(3)
    half = delta.where(ok & delta.between(0, 1.8))
    strict = raw_target(frame)
    all_ten = strict.notna() & strict.shift(1).notna() & strict.shift(2).notna()
    provenance = np.where(half.notna(), np.where(all_ten,'three_observed_increments','observed_endpoints_only'), 'unavailable')
    return strict.to_numpy(float), half.to_numpy(float), provenance


def make_targets(frame):
    n = len(frame); ten, half, provenance = endpoint_labels(frame)
    targets = np.full((n,49), np.nan, dtype=np.float32)
    ends = np.tile(np.r_[1,np.arange(3,145,3)], (n,1))+np.arange(n)[:,None]
    targets[:-1,0] = ten[1:]
    for j,h in enumerate(range(3,145,3)):
        targets[:-h,j+1] = half[h:]/3
    return targets, ends, provenance


def plan_tokens(frame, issues, mode):
    """Only oracle accesses future observations. Tokens contain NO fuel/weather GT."""
    if mode not in MODES: raise ValueError(mode)
    issues = np.asarray(issues,dtype=int)
    n=len(frame); sog=frame.SOG.where(frame.SOG.between(0,50)).to_numpy(float)
    heading=frame.HEADING.where(frame.HEADING.between(0,360)).to_numpy(float)
    offsets=np.r_[1,np.arange(3,145,3)]
    result=np.zeros((len(issues),49,9),dtype=np.float32)
    for j,h in enumerate(offsets):
        duration = 1 if j==0 else 3
        # Historical synthetic plan covers exactly the preceding RECEIVED 24h.
        positions=issues[:,None]+np.arange(h-duration+1,h+1)[None,:]
        if mode==MODES[1]:positions=positions-147
        clipped=np.clip(positions,0,n-1);valid=(positions>=0)&(positions<n)
        if mode!=MODES[2]:
            if mode==MODES[1]: assert (positions <= issues[:,None]-3).all()
            else: valid[:]=False
        sv=np.where(valid,sog[clipped],np.nan)
        hd=np.where(valid,heading[clipped],np.nan)
        ms=np.isfinite(sv).all(1); mh=np.isfinite(hd).all(1)
        result[:,j,0]=np.where(ms,np.nan_to_num(sv).mean(1)/50,0)
        result[:,j,1]=np.where(mh,np.sin(np.deg2rad(np.nan_to_num(hd))).mean(1),0)
        result[:,j,2]=np.where(mh,np.cos(np.deg2rad(np.nan_to_num(hd))).mean(1),0)
        result[:,j,3]=ms;result[:,j,4]=mh
        result[:,j,5]=h/144
        dates=frame.index[np.minimum(issues+h,n-1)]
        hour=dates.hour.to_numpy()+dates.minute.to_numpy()/60
        result[:,j,6]=np.sin(2*np.pi*hour/24);result[:,j,7]=np.cos(2*np.pi*hour/24)
        result[:,j,8]=duration/144
    assert np.isfinite(result).all()
    return result


class PlanHead(nn.Module):
    """Frozen cached base context -> plan-conditioned reused GRPO head.

    Initial zero residual exactly reproduces old 10min and 30min outputs.
    This is one policy, not an ensemble or a full pretrained-FM fine-tune.
    """
    def __init__(self, legacy_head):
        super().__init__();self.head=copy.deepcopy(legacy_head)
        self.plan_conditioner=nn.Linear(9,80)
        nn.init.zeros_(self.plan_conditioner.weight);nn.init.zeros_(self.plan_conditioner.bias)

    def forward(self, context, tokens):
        hidden=context[:,None,:]+torch.tanh(self.plan_conditioner(tokens))
        logits=self.head(hidden)
        selected=torch.cat([logits[:,:1,0],logits[:,1:,1]],dim=1)
        return torch.nn.functional.softplus(selected)


class PreservedShortPlanHead(nn.Module):
    """One policy with short/long temporal readouts; no prediction ensemble.

    Offline fitting changes only the long readout. Online released labels may
    update both readouts using one optimizer and one reference policy.
    """
    def __init__(self, legacy_head):
        super().__init__()
        self.short_head=copy.deepcopy(legacy_head)
        self.long_head=copy.deepcopy(legacy_head)
        self.plan_conditioner=nn.Linear(9,80)
        nn.init.zeros_(self.plan_conditioner.weight);nn.init.zeros_(self.plan_conditioner.bias)

    def forward(self,context,tokens):
        short=torch.nn.functional.softplus(self.short_head(context))
        hidden=context[:,None,:]+torch.tanh(self.plan_conditioner(tokens[:,3:]))
        long=torch.nn.functional.softplus(self.long_head(hidden)[:,:,1])
        second=(2*short[:,2]-short[:,1]).clamp_min(0)
        return torch.cat([short[:,:2],second[:,None],long],dim=1)


class AnticipatingPlanHead(PreservedShortPlanHead):
    """Optional bounded near-plan residual; missing plan exactly recovers base.

    The old short and long tensors are preserved during adapter-only fitting.
    """
    def __init__(self,legacy_head):
        super().__init__(legacy_head)
        self.near_plan=nn.Sequential(nn.Linear(89,32),nn.Tanh(),nn.Linear(32,1))
        nn.init.zeros_(self.near_plan[-1].weight);nn.init.zeros_(self.near_plan[-1].bias)

    def forward(self,context,tokens):
        near=tokens[:,:3].clone()
        near[:,2,:3]=(tokens[:,1,:3]+tokens[:,2,:3])/2
        near[:,2,3:5]=torch.minimum(tokens[:,1,3:5],tokens[:,2,3:5])
        near[:,2,8]=6/144
        delta=.25*torch.tanh(self.near_plan(torch.cat([context[:,None,:].expand(-1,3,-1),near],dim=-1))).squeeze(-1)
        known=near[:,:,3]*near[:,:,4]
        short=torch.nn.functional.softplus(self.short_head(context)+delta*known)
        hidden=context[:,None,:]+torch.tanh(self.plan_conditioner(tokens[:,3:]))
        long=torch.nn.functional.softplus(self.long_head(hidden)[:,:,1])
        return torch.cat([short[:,:2],(2*short[:,2]-short[:,1]).clamp_min(0)[:,None],long],dim=1)


def objective(pred, target):
    mask=torch.isfinite(target)
    weights=torch.ones_like(target);weights[:,0]=12
    weights*=mask
    if not mask.any():raise ValueError('No released target')
    safe=torch.where(mask,target,0)
    per=torch.nn.functional.smooth_l1_loss(pred,safe,beta=.2,reduction='none')+.15*(pred-safe).square()
    return (per*weights).sum()/weights.sum()


def update_grpo(model,reference,batch,opt):
    context,tokens,truth=batch;valid=torch.isfinite(truth)
    weights=valid.float();weights[:,0]*=12
    with torch.no_grad():
        old=model(context,tokens);ref=reference(context,tokens)
        candidates=torch.distributions.Normal(old,.15).sample((8,))
        error=torch.where(valid[None],candidates-truth[None],0)
        reward=-((error.abs()+.15*error.square())*weights[None]).sum(-1)/weights.sum(-1).clamp_min(1)
        advantage=(reward-reward.mean(0))/reward.std(0,unbiased=False).clamp_min(1e-6)
        old_lp=(torch.distributions.Normal(old,.15).log_prob(candidates)*valid[None]).sum(-1)
    for _ in range(2):
        pred=model(context,tokens)
        lp=(torch.distributions.Normal(pred,.15).log_prob(candidates)*valid[None]).sum(-1)
        ratio=torch.exp((lp-old_lp).clamp(-20,20))
        pg=torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
        kl=(((pred-ref).square()/(2*.15**2))*weights).sum()/weights.sum()
        value=-pg+.05*kl+.5*objective(pred,truth)
        opt.zero_grad();value.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()


def split_issues(frame,split):
    a,b={'train':('2025-01-01','2026-09-11'),'validation':('2026-09-11','2026-09-21'),
         'test':('2026-09-21','2026-10-01')}[split]
    ids=np.flatnonzero((frame.index>=a)&(frame.index<b))
    return ids[(ids>=147)&(ids+144<np.searchsorted(frame.index,pd.Timestamp(b)))&(ids+144<len(frame))]


def short_outputs(pred):
    return np.c_[pred[:,0],pred[:,1],pred[:,1:3].mean(1)]


def short_truth(targets):
    out=short_outputs(targets)
    out[:,2]=np.where(np.isfinite(targets[:,1:3]).all(1),out[:,2],np.nan)
    return out
