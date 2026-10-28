"""CPU offline SFT -> GRPO base; outputs are private artifacts, never Git data."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from operational_core import Policy, SCALE, causal_channels, grpo_step, loss, raw_target, score
from replay_grpo_base_20261015 import features, horizons, ns_times, sha, state_sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--offline-grpo-calls', type=int, default=20)
    p.add_argument('--historical-selection', action='store_true',
                   help='Archived development selection; unpurged validation. NOT an independent-test protocol.')
    a = p.parse_args()
    assert a.epochs >= 5 and a.epochs % 5 == 0 and a.offline_grpo_calls >= 0
    assert a.out.resolve().is_relative_to(a.root.resolve())
    a.out.mkdir(parents=True, exist_ok=False); torch.set_num_threads(4); began = time.monotonic()
    source = a.root/'operational_qc_v3_regenerated_20261013/merged_qc_frame.csv.gz'
    cachepath = a.root/'fresh_fm_cache_20261013/fresh_dual_cache.npz'
    frame = pd.read_csv(source, parse_dates=['timestamp']).set_index('timestamp')
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    assert frame.index.to_series().diff().dropna().eq(pd.Timedelta(minutes=10)).all()
    cache = np.load(cachepath, allow_pickle=False)
    assert str(cache['source_sha']) == sha(source)
    assert np.array_equal(cache['issue_indices'], np.arange(14, len(frame), 6))
    train = np.flatnonzero(frame.index < '2026-09-11')
    validation = np.flatnonzero((frame.index >= '2026-09-11') & (frame.index < '2026-09-21'))
    target = raw_target(frame).to_numpy()
    y, _ = horizons(target, ns_times(frame.index))
    fit = train[(train+6 < validation[0]) & np.isfinite(y[train]).any(1)]
    selected_validation = validation if a.historical_selection else validation[validation+6 < validation[-1]+1]
    channel_scalers = []
    for values, _, _ in causal_channels(frame, 3):
        median = np.nan_to_num(np.nanmedian(values[train], axis=0))
        std = np.nanstd(values[train], axis=0); std = np.where(std > 1e-6, std, 1.)
        channel_scalers.append({'median': median.tolist(), 'std': std.tolist()})
    indices = cache['issue_indices']; positions = np.searchsorted(indices, np.arange(len(frame)), side='right')-1
    fm = cache['x'][np.maximum(positions, 0)].copy(); fm[positions < 0] = 0
    mean = fm[train].mean(0); std = fm[train].std(0); std = np.where(std > 1e-6, std, 1.)
    scalers = dict(channel_scalers=channel_scalers, fm_mean=mean, fm_std=std)
    torch.save(scalers, a.out/'scalers.pt')
    tensors = features(frame, scalers, cache); yt = torch.tensor(y/SCALE, dtype=torch.float32)
    def batch(ids): return (*[t[ids] for t in tensors], yt[ids])
    protocol = dict(offline_training_only=True, seeds=[13, 37, 71], epochs=a.epochs,
        offline_grpo_calls=a.offline_grpo_calls, actual_offline_grpo_optimizer_steps=2*a.offline_grpo_calls,
        historical_unpurged_selection=a.historical_selection, training_rows=len(fit),
        source_sha256=sha(source), cache_sha256=sha(cachepath), validation_selection_rows=len(selected_validation),
        no_test_selection=True, cached_pretrained_scope=True, no_deployment=True)
    reference_rows = []
    for seed in [13, 37, 71]:
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        model = Policy(*[t.shape[1] for t in tensors])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        best, state = float('inf'), None
        for epoch in range(a.epochs):
            if time.monotonic()-began > 21600: raise TimeoutError('Six-hour CPU budget exceeded; partial artifacts preserved')
            for ids in np.array_split(rng.permutation(fit), max(1, len(fit)//512)):
                v, e, fm, truth = batch(ids); value = loss(model(v, e, fm), truth)
                optimizer.zero_grad(); value.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            if (epoch+1) % 5 == 0:
                with torch.no_grad(): pred = model(*batch(selected_validation)[:3]).numpy()*SCALE
                metric = score(y[selected_validation, 0], pred[:, 0])['mae']
                if metric < best: best, state = metric, copy.deepcopy(model.state_dict())
        model.load_state_dict(state); reference = copy.deepcopy(model); grpo = copy.deepcopy(model)
        for name, value in grpo.named_parameters(): value.requires_grad_(name.startswith('head.'))
        optimizer = torch.optim.Adam(grpo.head.parameters(), lr=3e-5)
        for _ in range(a.offline_grpo_calls):
            ids = rng.choice(fit, min(128, len(fit)), replace=False)
            grpo_step(grpo, reference, batch(ids), optimizer, 'accuracy')
        torch.save(dict(sft=model.state_dict(), grpo=grpo.state_dict(), protocol=protocol), a.out/f'checkpoint_seed{seed}.pt')
        # Covers the safely purged replay; archive-compatible column semantics.
        replay = validation[validation+6 < validation[-1]+1]
        for i in replay:
            with torch.no_grad(): value = grpo(*[t[i:i+1] for t in tensors]).numpy()[0, 0]*SCALE
            reference_rows.append(dict(seed=seed, arm='rolling_grpo', issued_at=str(frame.index[i]), prediction=float(value)))
        print(json.dumps(dict(seed=seed, best_development_mae=best, grpo_tensor_sha256=state_sha(grpo.state_dict()))), flush=True)
    pd.DataFrame(reference_rows).to_csv(a.out/'predictions.csv.gz', index=False, compression='gzip')
    (a.out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    manifest = dict(schema='kriso-cached-grpo-artifacts-v1', base_directory=str(a.out.resolve().relative_to(a.root.resolve())),
        source_sha256=sha(source), cache_sha256=sha(cachepath), scaler_sha256=sha(a.out/'scalers.pt'),
        checkpoint_sha256={seed: sha(a.out/f'checkpoint_seed{seed}.pt') for seed in [13, 37, 71]})
    (a.out/'artifacts.json').write_text(json.dumps(manifest, indent=2))
    print('BASE_TRAINING_COMPLETE')


if __name__ == '__main__': main()
