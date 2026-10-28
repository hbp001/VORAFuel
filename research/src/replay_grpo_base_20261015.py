"""Locked cached-FM offline-GRPO base, causal head-only online comparisons.

No offline retraining, deployment or test-informed selection. Existing September
periods are development/exploratory data, not an independent holdout.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from operational_core import (
    Policy, SCALE, VESSEL, WEATHER, causal_channels, grpo_step, loss, raw_target, score,
)
from short_gap_labels import NATIVE, build_labels, release_times

BASE = 'results_operational_fresh_fm_20261013_strict_full'
SOURCE_SHA = '6ca7ef59d57f24bce56e2858d16993fe3bf5874ed67234e7d66cbdb0f7c3e69d'
SCALER_SHA = '73c33624c2c5548e90189607fc7178052eaa22abb7e138e3dcca849b4eaf03ec'
CACHE_SHA = 'b69cda2b9931cc3144a4db7e75206fb573a38d166b0231fe7f369f8d5c796f80'
CHECKPOINT_SHA = {
    13: '83c5ce8af43da00c1e7d9bb2f239dcf301453f5c33277cfba12d24860b593734',
    37: '3a95a65ee0fece19eebd745ae398a6ffb1e84fdf0f6196a6283c9afe0ff21c48',
    71: '43615a89fc4cc6408d845fb40eb6167e0830f6a5a1ba2ffb7d178d20934602e6',
}
ARMS = ['GRPO기본_가중치고정', 'GRPO기본_온라인SFT', 'GRPO기본_온라인GRPO']
NAT = np.iinfo(np.int64).max


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def ns_times(values):
    """Pandas may default to us; never mix asi8 with Timestamp.value (ns)."""
    return np.asarray(values, dtype='datetime64[ns]').view(np.int64)


def horizons(values, releases):
    """At issue i, each target is the mean of i+1 through i+h, never past."""
    values = np.asarray(values, float)
    releases = np.asarray(releases, dtype=np.int64)
    y = np.full((len(values), 3), np.nan)
    r = np.full((len(values), 3), NAT, dtype=np.int64)
    for j, h in enumerate([1, 3, 6]):
        vv = np.stack([values[k:len(values)-h+k] for k in range(1, h+1)], axis=1)
        rr = np.stack([releases[k:len(values)-h+k] for k in range(1, h+1)], axis=1)
        ok = np.isfinite(vv).all(1) & (rr != NAT).all(1)
        pos = np.flatnonzero(ok)
        y[pos, j] = vv[ok].mean(1); r[pos, j] = rr[ok].max(1)
    return y, r


def label_bundle(frame, track, blocked=None):
    labels = build_labels(frame)
    pd.testing.assert_series_equal(labels.strict_observed, raw_target(frame), check_names=False)
    delay = 1440 if track == 'QC실측24시간' else 30
    if track == 'QC실측24시간':
        values = frame.qc_target_observed.to_numpy().copy()
        release = ns_times(frame.index) + pd.Timedelta(minutes=delay).value
    elif track == '엄격실측30분':
        values = labels.strict_observed.to_numpy().copy()
        release = ns_times(frame.index) + pd.Timedelta(minutes=delay).value
    else:
        values = labels.with_short_interpolation.to_numpy().copy()
        release = ns_times(release_times(labels, delay)).copy()
    release[~np.isfinite(values)] = NAT
    finite = np.isfinite(values)
    assert (release[finite] >= ns_times(frame.index)[finite]+pd.Timedelta(minutes=delay).value).all()
    if blocked is not None:
        values[blocked] = np.nan; release[blocked] = NAT
    return (*horizons(values, release), labels)


def features(frame, scalers, cache, blocked_groups=None):
    tensors = []
    blocks = causal_channels(frame, 3)
    for (values, valid, ages), scaler in zip(blocks, scalers['channel_scalers']):
        z = np.nan_to_num((values-np.array(scaler['median']))/np.array(scaler['std'])).clip(-10, 10)
        z = np.concatenate([z, valid, ages], axis=1)
        windows = np.stack([np.roll(z, lag, axis=0) for lag in range(12)], axis=1)
        for lag in range(12):
            windows[:lag, lag] = 0
        tensors.append(torch.tensor(windows.reshape(len(frame), -1), dtype=torch.float32))
    issue_indices = cache['issue_indices']
    positions = np.searchsorted(issue_indices, np.arange(len(frame)), side='right')-1
    available = positions >= 0; pick = np.maximum(positions, 0)
    assert (issue_indices[pick[available]] <= np.flatnonzero(available)).all()
    fm = cache['x'][pick].copy(); flags = cache['group_present'][pick].copy()
    fm[~available] = 0; flags[~available] = False
    if blocked_groups is not None:
        for g in range(2):
            bad = blocked_groups[pick, g]
            fm[bad, g*768:(g+1)*768] = 0; flags[bad, g] = False
    age = np.where(available, (np.arange(len(frame))-issue_indices[pick])/6., 1.)[:, None]
    zfm = np.c_[((fm-scalers['fm_mean'])/scalers['fm_std']).clip(-10, 10), flags, age]
    tensors.append(torch.tensor(zfm, dtype=torch.float32))
    return tensors


class CountAdam(torch.optim.Adam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.actual_steps = 0

    def step(self, *args, **kwargs):
        self.actual_steps += 1
        return super().step(*args, **kwargs)


def self_test():
    values = np.arange(20, dtype=float)/100
    release = np.arange(20, dtype=np.int64)*10
    y, r = horizons(values, release)
    assert np.isclose(y[3, 2], values[4:10].mean()) and r[3, 2] == 90
    values[7] = np.nan; yy, rr = horizons(values, release)
    assert np.isnan(yy[3, 2]) and rr[3, 2] == NAT
    # Future labels are never passed through as unseen multi-horizon targets.
    cutoff = 80
    values2 = values.copy(); values2[release > cutoff] += 100
    y2, r2 = horizons(values2, release)
    mask = rr <= cutoff
    assert np.allclose(yy[mask], y2[mask], equal_nan=True)
    torch.manual_seed(13)
    model = Policy(3, 3, 3); reference = copy.deepcopy(model)
    for n, p in model.named_parameters(): p.requires_grad_(n.startswith('head.'))
    opt = CountAdam(model.head.parameters(), lr=3e-5)
    batch = (torch.ones(8, 3),)*3 + (torch.ones(8, 3),)
    before = copy.deepcopy(model.state_dict()); before_opt = copy.deepcopy(opt.state_dict())
    grpo_step(model, reference, batch, opt, 'accuracy')
    assert opt.actual_steps == 2
    model.load_state_dict(before); opt.load_state_dict(before_opt)
    assert state_sha(model.state_dict()) == state_sha(before)
    return {'horizon_alignment': True, 'future_release_mask': True,
            'grpo_actual_two_steps': True, 'bitexact_rollback': True}


def gap_frame(original, replay, duration, mode, track):
    """Evaluation truth stays outside this modified learning/input frame."""
    frame = original.copy(); steps = duration//10
    hidden = np.zeros(len(frame), bool); recovery = hidden.copy(); starts = []
    truth = raw_target(original).to_numpy() if track != 'QC실측24시간' else original.qc_target_observed.to_numpy()
    stationary = (original.SOG.to_numpy() < .5) & np.isfinite(truth)
    i = replay[0]+1
    while i+steps+12 < replay[-1] and len(starts) < 12:
        if stationary[i:i+steps].all():
            starts.append(int(i)); hidden[i:i+steps] = True; recovery[i+steps:i+steps+12] = True
            i += steps+12
        else:
            i += 1
    columns = ['cam2_total', 'cam2_hfo', 'cam2_mdo']
    if mode != '연료계측공백':
        columns += [c for c in NATIVE if c in frame] + ['mean_rpm', 'mean_pitch']
    if mode == '전체입력공백': columns += WEATHER
    frame.loc[hidden, list(set(columns))] = np.nan
    # Counter at first post-gap interval shares hidden left endpoint. QC centered
    # selection can depend on nearby hidden counters; exclude its whole bound.
    blocked = hidden.copy()
    for start in starts:
        lo = max(0, start-(40 if track == 'QC실측24시간' else 0))
        hi = min(len(frame), start+steps+1+(40 if track == 'QC실측24시간' else 0))
        blocked[lo:hi] = True
    return frame, hidden, recovery, blocked, starts


def blocked_cache(original, modified, cache):
    """Conservative cache fallback after actual affected context/ffill changes.

    Compare the exact delayed, past-filled raw Chronos input arrays. If any
    channel differs anywhere in its 12-step context, disable that group for this
    cache issue. This covers right-boundary raw-difference and stale carry too.
    """
    old, new = causal_channels(original, 3), causal_channels(modified, 3)
    out = np.zeros((len(cache['issue_indices']), 2), bool)
    for g in range(2):
        unequal = (~np.isclose(old[g][0], new[g][0], equal_nan=True).all(1)
                   | (old[g][1] != new[g][1]).any(1))
        for j, i in enumerate(cache['issue_indices']): out[j, g] = unequal[i-11:i+1].any()
    return out


def run(args):
    out = args.out; out.mkdir(parents=True, exist_ok=False)
    root = args.root; started = time.time(); torch.set_num_threads(4)
    checks = self_test()
    expected_source, expected_cache, expected_scaler = SOURCE_SHA, CACHE_SHA, SCALER_SHA
    expected_checkpoints, base_directory = CHECKPOINT_SHA, BASE
    if args.artifacts_manifest:
        artifacts = json.loads(args.artifacts_manifest.read_text())
        if artifacts.get('schema') != 'kriso-cached-grpo-artifacts-v1':
            raise ValueError('Unsupported artifacts manifest')
        base_directory = artifacts['base_directory']
        if Path(base_directory).is_absolute() or '..' in Path(base_directory).parts:
            raise ValueError('base_directory must be a safe relative path under --root')
        expected_source = artifacts['source_sha256']; expected_cache = artifacts['cache_sha256']
        expected_scaler = artifacts['scaler_sha256']
        expected_checkpoints = {int(k): v for k, v in artifacts['checkpoint_sha256'].items()}
        assert set(expected_checkpoints) == {13, 37, 71}
    source = root/'operational_qc_v3_regenerated_20261013/merged_qc_frame.csv.gz'
    cache_path = root/'fresh_fm_cache_20261013/fresh_dual_cache.npz'
    scaler_path = root/base_directory/'scalers.pt'
    assert sha(source) == expected_source and sha(cache_path) == expected_cache and sha(scaler_path) == expected_scaler
    frame = pd.read_csv(source, parse_dates=['timestamp']).set_index('timestamp')
    original = frame.copy(); cache = np.load(cache_path)
    assert np.array_equal(cache['issue_indices'], np.arange(14, len(frame), 6))
    scalers = torch.load(scaler_path, weights_only=False)
    train = np.flatnonzero(frame.index < '2026-09-11')
    start, end = ('2026-09-11', '2026-09-21') if args.split == 'validation' else ('2026-09-21', '2026-10-01')
    replay = np.flatnonzero((frame.index >= start) & (frame.index < end))
    purge = 48 if args.track == 'QC실측24시간' else 6
    replay = replay[replay+purge < np.searchsorted(frame.index, pd.Timestamp(end))]
    if args.pilot: replay = replay[:180]
    observed_values = (original.qc_target_observed if args.track == 'QC실측24시간' else raw_target(original)).to_numpy()
    observed, _ = horizons(observed_values, np.where(np.isfinite(observed_values), ns_times(original.index), NAT))
    augmented, _, _ = label_bundle(original, '엄격보간30분')
    hidden = np.zeros(len(frame), bool); recovery = hidden.copy(); starts = []; blocked_groups = None
    if args.gap_minutes:
        frame, hidden, recovery, blocked, starts = gap_frame(original, replay, args.gap_minutes, args.gap_mode, args.track)
        if not starts:
            save_json(out/'completion.json', {'complete': True, 'status': 'N_A', 'reason': 'No fully observed stationary window of specified duration'})
            return
        blocked_groups = blocked_cache(original, frame, cache)
    else: blocked = None
    if args.weather_unavailable:
        # Only replay-time weather reception is disabled; historical training
        # inputs and offline checkpoints/scalers are not changed.
        frame = frame.copy()
        frame.loc[frame.index[replay], WEATHER] = np.nan
        blocked_groups = blocked_cache(original, frame, cache)
    y, releases, labels = label_bundle(frame, args.track, blocked)
    tensors = features(frame, scalers, cache, blocked_groups)
    cut = replay[0]+73
    for (full, part) in zip(causal_channels(frame, 3), causal_channels(frame.iloc[:cut+1], 3)):
        for a, b in zip(full, part): assert np.allclose(a[:cut+1], b, equal_nan=True)
    # Whole-label future perturbation check after per-horizon release masking.
    now = frame.index[cut].value; yy = y.copy(); yy[releases > now] = 123456.
    assert np.allclose(np.where(releases <= now, y, np.nan), np.where(releases <= now, yy, np.nan), equal_nan=True)
    # Historical labels must be released before the replay starts; training only.
    fit = train[(train+purge < np.searchsorted(frame.index, pd.Timestamp('2026-09-11'))) & np.isfinite(y[train]).any(1)]
    guard = fit[::max(1, len(fit)//256)][:256]
    retained = np.setdiff1d(fit[::19], guard)
    historical_pool = np.setdiff1d(fit, retained)
    assert len(historical_pool) >= 64 and not np.intersect1d(historical_pool, retained).size
    yt = torch.tensor(y/SCALE, dtype=torch.float32)
    update_every, steps = (36, 12) if args.track == 'QC실측24시간' else (6, 2)
    protocol = dict(name='캐시형 GRPO기본 이후 인과 온라인 업데이트', split=args.split, track=args.track,
        source_sha256=expected_source, cache_sha256=expected_cache, scaler_sha256=expected_scaler,
        checkpoint_sha256=expected_checkpoints, initial_key='grpo', seeds=[13, 37, 71],
        artifact_origin='explicit manifest' if args.artifacts_manifest else 'locked historical reference',
        timestamp_unit='explicit datetime64[ns] for every release and issuance', version=2,
        weather_unavailable=bool(args.weather_unavailable),
        training='NO offline retraining; same existing offline GRPO in all arms/tracks',
        scope='cached frozen raw Chronos dual-stream + downstream dual encoders/single head; only head online',
        evaluation='development validation / previously inspected exploratory test; NO independent or real-vessel claim',
        label_delay_minutes=1440 if args.track == 'QC실측24시간' else 30, input_delay_minutes=30,
        update_minutes=update_every*10, actual_steps_per_update=steps, reward='accuracy preregistered',
        horizon_minutes=[10, 30, 60], units='MT/10min horizon mean; not horizon total',
        purge_steps=purge, replay_rows=len(replay), guard='training smoothL1+.15MSE <=1.05 previous; full model+optimizer rollback',
        retention='offline-training probe excluded from online buffer, NOT heldout generalization',
        gap_minutes=args.gap_minutes, gap_mode=args.gap_mode, gap_start_indices=starts,
        blocked_cache_groups=int(blocked_groups.sum()) if blocked_groups is not None else 0,
        missingness='only approved 1/2 counter-sample labels interpolate after right anchor; no input interpolation change',
        uncertainty='training-observed retained residual90%, adaptive released prediction residual recent144 min30; diagnostic no coverage guarantee',
        unavailable='independent vessel/period; actual berth fuel/power truth; real-vessel deployment; actual weather available_at',
        deployment=False, checks=checks)
    save_json(out/'protocol.json', protocol)
    metrics, predictions, updates, retention_rows, initial_rows = [], [], [], [], []
    for seed in ([13] if args.pilot else [13, 37, 71]):
        cp = root/base_directory/f'checkpoint_seed{seed}.pt'; assert sha(cp) == expected_checkpoints[seed]
        saved = torch.load(cp, weights_only=False)
        model = Policy(*[t.shape[1] for t in tensors]); model.load_state_dict(saved['grpo']); model.eval()
        reference = copy.deepcopy(model); initial_hash = state_sha(model.state_dict())
        with torch.no_grad():
            initial_prediction = model(*[t[retained] for t in tensors]).numpy()*SCALE
            guard_truth = torch.tensor(observed[guard]/SCALE, dtype=torch.float32)
        residual = np.abs(initial_prediction[:, 0]-observed[retained, 0])
        residual = residual[np.isfinite(residual)]; fixed_q = float(np.quantile(residual, .9, method='higher'))
        # Prove original scaler, ordering and checkpoint reproduce saved base.
        if not args.gap_minutes and not args.weather_unavailable and args.split == 'validation':
            old = pd.read_csv(root/base_directory/'predictions.csv.gz')
            old = old[(old.seed == seed) & (old.arm == 'rolling_grpo')].set_index('issued_at')
            with torch.no_grad(): check = model(*[t[replay] for t in tensors]).numpy()[:, 0]*SCALE
            expected = old.loc[frame.index[replay].astype(str), 'prediction'].to_numpy()
            assert np.allclose(check, expected, atol=2e-7, rtol=0), np.max(abs(check-expected))
        frozen_nonhead = state_sha({k: v for k, v in reference.state_dict().items() if not k.startswith('head.')})
        for arm in ARMS:
            active = copy.deepcopy(model)
            assert state_sha(active.state_dict()) == initial_hash
            initial_rows.append(dict(seed=seed, arm=arm, tensor_sha256=initial_hash))
            for n, p in active.named_parameters(): p.requires_grad_(n.startswith('head.'))
            opt = CountAdam(active.head.parameters(), lr=3e-5)
            arm_rng = np.random.default_rng(seed+13000); torch.manual_seed(seed+13000)
            pred = np.full((len(replay), 3), np.nan); fixed_interval = []; adaptive_interval = []
            released_prediction = set(); residual_buffer = []; arm_started = time.perf_counter()
            for k, i in enumerate(replay):
                now = frame.index[i].value
                # Each issued prediction contributes at most once, only on release.
                ready = np.flatnonzero((releases[replay[:k], 0] <= now) & np.isfinite(y[replay[:k], 0]))
                for kk in ready:
                    if int(kk) not in released_prediction:
                        residual_buffer.append(abs(pred[kk, 0]-y[replay[kk], 0])); released_prediction.add(int(kk))
                if arm != ARMS[0] and k > 0 and k % update_every == 0:
                    available = replay[:k][np.any((releases[replay[:k]] <= now) & np.isfinite(y[replay[:k]]), axis=1)]
                    if len(available) >= 16:
                        ids = np.r_[available[-144:], arm_rng.choice(historical_pool, 64, replace=False)]
                        mask = (releases[ids] <= now) & np.isfinite(y[ids])
                        truth = yt[ids].clone(); truth[~torch.tensor(mask)] = float('nan')
                        assert np.isfinite(truth.numpy()).any(1).all()
                        assert not np.any(np.isfinite(truth.numpy()) & (releases[ids] > now))
                        before = copy.deepcopy(active.state_dict()); opt_before = copy.deepcopy(opt.state_dict())
                        with torch.no_grad(): oldloss = loss(active(*[t[guard] for t in tensors]), guard_truth).item()
                        b = (*[t[ids] for t in tensors], truth)
                        n0 = opt.actual_steps; tick = time.perf_counter()
                        for _ in range(steps//2 if arm == ARMS[2] else steps):
                            if arm == ARMS[2]: grpo_step(active, reference, b, opt, 'accuracy')
                            else:
                                value = loss(active(*b[:3]), b[3]); opt.zero_grad(); value.backward()
                                nn.utils.clip_grad_norm_(active.head.parameters(), 1.); opt.step()
                        actual_steps = opt.actual_steps-n0; assert actual_steps == steps
                        with torch.no_grad(): newloss = loss(active(*[t[guard] for t in tensors]), guard_truth).item()
                        accepted = bool(np.isfinite(newloss) and newloss <= oldloss*1.05)
                        if not accepted:
                            active.load_state_dict(before); opt.load_state_dict(opt_before)
                            assert state_sha(active.state_dict()) == state_sha(before)
                        updates.append(dict(seed=seed, arm=arm, issued_at=str(frame.index[i]), accepted=accepted,
                            guard_before=oldloss, guard_after=newloss, actual_steps=actual_steps,
                            buffer_sha256=hashlib.sha256(ids.astype(np.int64).tobytes()+mask.tobytes()).hexdigest(),
                            latest_released_at=str(pd.Timestamp(int(releases[ids][mask].max()))),
                            training_label_count=int(mask.sum()), update_seconds=time.perf_counter()-tick))
                with torch.no_grad(): pred[k] = active(*[t[i:i+1] for t in tensors]).numpy()[0]*SCALE
                fixed_interval.append(fixed_q)
                adaptive_interval.append(float(np.quantile(residual_buffer[-144:], .9, method='higher')) if len(residual_buffer) >= 30 else fixed_q)
            assert np.isfinite(pred).all() and (pred >= 0).all()
            assert state_sha({k: v for k, v in active.state_dict().items() if not k.startswith('head.')}) == frozen_nonhead
            arm_seconds = time.perf_counter()-arm_started
            with torch.no_grad(): final_retained = active(*[t[retained] for t in tensors]).numpy()*SCALE
            for j, h in enumerate([10, 30, 60]):
                retention_rows.append(dict(seed=seed, arm=arm, horizon_min=h,
                    initial_mae=score(observed[retained, j], initial_prediction[:, j])['mae'],
                    **score(observed[retained, j], final_retained[:, j])))
            sog = original.SOG.to_numpy(); ds = original.SOG.diff().to_numpy()
            turn = ((original.HEADING.diff()+180)%360-180).abs().to_numpy()
            segments = {'전체': np.ones(len(replay), bool), '정지proxy': sog[replay] < .5,
                '항행proxy': sog[replay] >= 5, '속도불명': ~np.isfinite(sog[replay]),
                '급속도변화proxy': abs(ds[replay]) >= 2, '회전proxy': (turn[replay] >= 15) & (sog[replay] >= 5),
                '숨김': hidden[replay+1], '복귀': recovery[replay+1], '보통': ~(hidden[replay+1] | recovery[replay+1])}
            for segment, m in segments.items():
                for j, h in enumerate([10, 30, 60]):
                    metrics.append(dict(seed=seed, arm=arm, segment=segment, horizon_min=h,
                        **score(observed[replay[m], j], pred[m, j]), arm_seconds=arm_seconds))
            metrics.append(dict(seed=seed, arm=arm, segment='엄격보간포함보조', horizon_min=10,
                **score(augmented[replay, 0], pred[:, 0]), arm_seconds=arm_seconds))
            for k, i in enumerate(replay):
                for j, h in enumerate([10, 30, 60]):
                    predictions.append(dict(seed=seed, arm=arm, issued_at=str(frame.index[i]), horizon_min=h,
                        label_end=str(frame.index[i+h//10]), gt=observed[i, j], prediction=pred[k, j],
                        auxiliary_gt=augmented[i, j], training_target=y[i, j],
                        training_release_at=str(pd.Timestamp(int(releases[i, j]))) if releases[i, j] != NAT else '',
                        hidden=bool(hidden[i+1:i+h//10+1].any()), recovery=bool(recovery[i+1]),
                        fixed_q=fixed_interval[k] if j == 0 else np.nan,
                        adaptive_q=adaptive_interval[k] if j == 0 else np.nan))
            torch.save({'state_dict': active.state_dict(), 'initial_tensor_sha256': initial_hash,
                        'protocol': protocol}, out/f'final_seed{seed}_{arm}.pt')
            print(json.dumps(dict(seed=seed, arm=arm, seconds=arm_seconds,
                **score(observed[replay, 0], pred[:, 0])), ensure_ascii=False), flush=True)
        for name, data in [('metrics', metrics), ('updates', updates), ('retention', retention_rows), ('initial', initial_rows)]:
            pd.DataFrame(data).to_csv(out/f'{name}.csv', index=False)
        pd.DataFrame(predictions).to_csv(out/'predictions.csv.gz', index=False, compression='gzip')
    ud = pd.DataFrame(updates)
    if len(ud):
        one = ud[ud.arm == ARMS[1]].reset_index(drop=True)
        two = ud[ud.arm == ARMS[2]].reset_index(drop=True)
        for col in ['seed', 'issued_at', 'actual_steps', 'buffer_sha256', 'latest_released_at', 'training_label_count']:
            assert one[col].equals(two[col]), ('unmatched', col)
    save_json(out/'completion.json', dict(complete=True, elapsed_seconds=time.time()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, initial_equality=True,
        frozen_nonhead_unchanged=True, actual_steps_matched=True, buffer_release_matched=True,
        no_deployment=True, no_offline_retraining=True, baseline_saved_prediction_match=not args.gap_minutes and not args.weather_unavailable and args.split == 'validation',
        files={p.name: sha(p) for p in out.iterdir() if p.is_file()}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True, help='Private artifact workspace; no data are bundled')
    p.add_argument('--artifacts-manifest', type=Path, help='Explicit hashes for a newly reproduced base; default locks the archived reference')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--track', choices=['엄격실측30분', '엄격보간30분', 'QC실측24시간'], required=True)
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--weather-unavailable', action='store_true')
    p.add_argument('--gap-minutes', type=int, choices=[30, 120, 360], default=0)
    p.add_argument('--gap-mode', choices=['연료계측공백', '운항포함공백', '전체입력공백'], default='연료계측공백')
    a = p.parse_args()
    lock = (a.root/'grpo_base_replay_20261015.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run(a)
