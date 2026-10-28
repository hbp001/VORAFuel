"""Cached dual-stream fuel policy and the evaluated accuracy-GRPO update.

Extracted without equation/parameter-order changes from the archived experiment.
The foundation model is frozen and cached; this is not an end-to-end trained FM.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

VESSEL = ['SOG', 'mean_rpm', 'mean_pitch', 'PORT_ENGINE_STATUS',
          'STBD_ENGINE_STATUS', 'heading_sin', 'heading_cos', 'raw_fuel']
WEATHER = ['kma_fc_wave_m', 'kma_fc_period_s', 'kma_fc_wind_u',
           'kma_fc_wind_v', 'kma_fc_wave_dir_sin', 'kma_fc_wave_dir_cos']
SCALE = .1


def raw_target(frame):
    total = frame.cam2_total.where(frame.cam2_total.between(10000, 100000))
    parts = (frame.cam2_hfo+frame.cam2_mdo).where((frame.cam2_hfo+frame.cam2_mdo).between(10000, 100000))
    level = pd.concat([total, parts], axis=1).median(axis=1, skipna=True)
    agreement = total.notna() & parts.notna() & ((total-parts).abs() <= .02)
    delta = level.diff()
    return delta.where(delta.between(0, .6) & agreement & agreement.shift(1, fill_value=False))


def causal_channels(frame, delay):
    frame = frame.copy()
    frame['raw_fuel'] = raw_target(frame)
    heading = frame.HEADING.where(frame.HEADING.between(0, 360))
    frame['heading_sin'] = np.sin(np.deg2rad(heading))
    frame['heading_cos'] = np.cos(np.deg2rad(heading))
    frame['SOG'] = frame.SOG.where(frame.SOG.between(0, 50))
    for column in ['PORT_ENGINE_STATUS', 'STBD_ENGINE_STATUS']:
        frame[column] = frame[column].where(frame[column].isin([1, 2]))
    blocks = []
    for columns in [VESSEL, WEATHER]:
        values = frame[columns].shift(delay)
        valid = values.notna().to_numpy()
        positions = np.arange(len(frame))[:, None]
        last = np.maximum.accumulate(np.where(valid, positions, -100000), axis=0)
        ages = np.minimum(positions-last, 144)/144.
        blocks.append((values.ffill(limit=12).to_numpy(float), valid.astype(float), ages))
    return blocks


class Policy(nn.Module):
    def __init__(self, nv, ne, nf):
        super().__init__()
        self.vessel = nn.Sequential(nn.Linear(nv, 32), nn.Tanh())
        self.weather = nn.Sequential(nn.Linear(ne, 16), nn.Tanh())
        self.fm = nn.Sequential(nn.Linear(nf, 32), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(80, 32), nn.Tanh(), nn.Linear(32, 3))

    def forward(self, vessel, environment, fm):
        return torch.nn.functional.softplus(self.head(torch.cat([
            self.vessel(vessel), self.weather(environment), self.fm(fm)], dim=1)))


def loss(prediction, target):
    mask = torch.isfinite(target)
    if not mask.any():
        raise ValueError('No released finite label in this batch')
    return (torch.nn.functional.smooth_l1_loss(prediction[mask], target[mask], beta=.2)
            + .15*((prediction[mask]-target[mask])**2).mean())


def grpo_step(model, reference, batch, optimizer, mode='accuracy', context=None):
    """One group sampling call performs TWO actual optimizer steps.

    Eight Normal candidates; group-relative advantages; clipped likelihood ratio;
    exact same-variance Gaussian reference KL plus a supervised anchor. This is
    the evaluated regression policy, not a claim to all variants of LLM GRPO.
    """
    if mode != 'accuracy' or context is not None:
        raise ValueError('This release implements the evaluated accuracy reward only')
    vessel, environment, fm, target = batch
    with torch.no_grad():
        old = model(vessel, environment, fm)
        ref = reference(vessel, environment, fm)
        candidates = torch.distributions.Normal(old, .15).sample((8,)).detach()
        mask = torch.isfinite(target)[None]
        error = torch.where(mask, candidates-target[None], 0)
        reward = -(error.abs()+.15*error.square()).sum(-1)/mask.sum(-1).clamp_min(1)
        advantage = (reward-reward.mean(0))/reward.std(0, unbiased=False).clamp_min(1e-6)
        old_lp = torch.distributions.Normal(old, .15).log_prob(candidates).sum(-1)
    for _ in range(2):
        current = model(vessel, environment, fm)
        new_lp = torch.distributions.Normal(current, .15).log_prob(candidates).sum(-1)
        ratio = torch.exp((new_lp-old_lp).clamp(-20, 20))
        objective = torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage).mean()
        kl = ((current-ref).square()/(2*.15**2)).sum(-1).mean()
        value = -objective+.05*kl+.5*loss(current, target)
        optimizer.zero_grad(); value.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()


def score(target, prediction):
    valid = np.isfinite(target) & np.isfinite(prediction)
    if not valid.any():
        return {'n': 0, 'mae': None, 'rmse': None, 'p95': None, 'bias': None}
    delta = prediction[valid]-target[valid]
    return dict(n=int(valid.sum()), mae=float(np.abs(delta).mean()),
                rmse=float(np.sqrt((delta*delta).mean())), bias=float(delta.mean()),
                p95=float(np.quantile(np.abs(delta), .95)))
