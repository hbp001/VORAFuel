"""Versioned short counter-gap labels; never alter model inputs or strict GT."""
from __future__ import annotations

import numpy as np
import pandas as pd

NATIVE = ['SOG', 'STW', 'HEADING', 'PORT_PITCH_OUTPUT', 'STBD_PITCH_OUTPUT',
          'PORT_RPM_OUTPUT', 'STBD_RPM_OUTPUT', 'PORT_ENGINE_STATUS', 'STBD_ENGINE_STATUS']


def build_labels(frame: pd.DataFrame, max_missing: int = 2) -> pd.DataFrame:
    if max_missing not in (1, 2):
        raise ValueError('Only preregistered one/two missing counter samples allowed')
    idx = frame.index
    if not isinstance(idx, pd.DatetimeIndex) or not idx.is_unique or not idx.is_monotonic_increasing:
        raise ValueError('Unique sorted DatetimeIndex required')
    if len(idx) > 1 and not (idx.to_series().diff().iloc[1:] == pd.Timedelta(minutes=10)).all():
        raise ValueError('Exact ten-minute grid required')
    total = frame.cam2_total.where(frame.cam2_total.between(10000, 100000))
    parts = (frame.cam2_hfo + frame.cam2_mdo).where((frame.cam2_hfo + frame.cam2_mdo).between(10000, 100000))
    raw_level = pd.concat([total, parts], axis=1).median(axis=1, skipna=True)
    agree = total.notna() & parts.notna() & ((total-parts).abs() <= .02)
    delta = raw_level.diff()
    strict = delta.where(delta.between(0, .6) & agree & agree.shift(1, fill_value=False))
    result = pd.DataFrame(index=idx)
    result['strict_observed'] = strict
    result['with_short_interpolation'] = strict
    result['label_type'] = np.where(strict.notna(), 'observed', 'unavailable')
    result['left_anchor'] = pd.NaT
    result['right_anchor'] = pd.NaT
    result['missing_counter_samples'] = 0
    # Genuine absent counters only: do not replace a QC-rejected observed value.
    absent = frame[['cam2_total', 'cam2_hfo', 'cam2_mdo']].isna().all(axis=1).to_numpy()
    sensor = frame[[c for c in NATIVE if c in frame]].notna().any(axis=1).to_numpy()
    # Never bridge fixed train/validation/test boundaries.
    split = np.searchsorted(pd.to_datetime(['2026-09-11', '2026-09-21', '2026-10-01']).values,
                            idx.values, side='right')
    i = 0
    while i < len(idx):
        if not absent[i]:
            i += 1
            continue
        start = i
        while i < len(idx) and absent[i]:
            i += 1
        stop = i  # first observed counter after run
        n = stop-start
        if n > max_missing or start == 0 or stop == len(idx):
            continue
        left = start-1
        if not (sensor[start:stop].all() and agree.iloc[left] and agree.iloc[stop]
                and split[left] == split[stop]):
            continue
        step = float((raw_level.iloc[stop]-raw_level.iloc[left])/(n+1))
        if not 0 <= step <= .6:
            continue
        # The right-anchor interval also depends on an interpolated counter.
        positions = np.arange(start, stop+1)
        result.iloc[positions, result.columns.get_loc('with_short_interpolation')] = step
        result.iloc[positions, result.columns.get_loc('label_type')] = 'short_gap_interpolated'
        result.loc[idx[positions], 'left_anchor'] = idx[left]
        result.loc[idx[positions], 'right_anchor'] = idx[stop]
        result.loc[idx[positions], 'missing_counter_samples'] = n
    return result


def release_times(labels: pd.DataFrame, label_delay_minutes: int,
                  receipt_delay_minutes: int = 30) -> pd.Series:
    if min(label_delay_minutes, receipt_delay_minutes) < 0:
        raise ValueError('Negative release delay')
    delay = pd.Timedelta(minutes=max(label_delay_minutes, receipt_delay_minutes))
    end = pd.Series(labels.index, index=labels.index)
    release = end + delay
    imputed = labels.label_type.eq('short_gap_interpolated')
    release.loc[imputed] = labels.loc[imputed, 'right_anchor'] + delay
    return release.where(labels.with_short_interpolation.notna())
