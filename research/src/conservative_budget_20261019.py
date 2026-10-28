"""Bounded single-path calibration of an entirely frozen GRPO policy.

No future actual state/target enters features or forward. This is supervised
residual calibration, NOT another GRPO comparison or an ensemble.
"""
import torch
from torch import nn

def features(base,raw,plans,recent):
    n=base.shape[1];r=raw[:,None].expand(-1,n,-1);z=recent[:,None].expand(-1,n,-1)
    h=torch.arange(1,n+1,dtype=base.dtype,device=base.device)[None].expand_as(base)/n
    return torch.stack([torch.ones_like(base),r[:,:,0],z[:,:,0],base/.3,
        plans[:,:,0]*plans[:,:,3],(plans[:,:,0]-r[:,:,0])*plans[:,:,3],h,h.square(),
        z[:,:,1],torch.exp(-(r[:,:,0]*50/3).square())*r[:,:,9],
        (plans[:,:,1]*r[:,:,2]+plans[:,:,2]*r[:,:,3])*plans[:,:,4]*r[:,:,11],plans[:,:,3]],-1)

class ConservativeResidual(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.zeros(12))
    def forward(self,base,x,alpha=1.):
        cap=.15*base+.001
        return (base+float(alpha)*cap*torch.tanh((x*self.weight).sum(-1))).clamp_min(0)

def loss(model,base,x,target,cohorts,total_weight):
    pred=model(base,x);valid=torch.isfinite(target);safe=torch.nan_to_num(target)
    err=(pred-safe).abs();old=(base-safe).abs();objective=err[valid].mean()/.3
    full=valid.all(1)
    if full.any():objective=objective+total_weight*(pred[full].sum(1)-target[full].sum(1)).abs().mean()/3.6
    for keep in [valid]+[valid&v for v in cohorts]:
        if keep.any():objective=objective+10*torch.relu(err[keep].mean()-old[keep].mean())/.3
    return objective+.05*((pred-base)/(.15*base+.001)).square().mean()
