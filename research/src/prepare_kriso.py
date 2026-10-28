#!/usr/bin/env python3
"""Build leakage-safe 10-minute and daily KRISO fuel-consumption datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_FEATURES = [
    "SOG", "STW", "HEADING", "DEPTH",
    "PORT_PITCH_OUTPUT", "PORT_RPM_OUTPUT", "PORT_ENGINE_STATUS",
    "STBD_PITCH_OUTPUT", "STBD_RPM_OUTPUT", "STBD_ENGINE_STATUS",
    "WIND_ANGLE", "WIND_SPEED", "htsgw", "dirpw", "perpw",
    "curx", "cury", "wx", "wy", "tmp",
]


def numeric_frame(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    for col in raw.columns:
        if col != "DateTime":
            out[col] = pd.to_numeric(raw[col], errors="coerce")
    return out


def _candidate_temporal_score(series: pd.Series) -> pd.Series:
    """Return a local linearity score for one cumulative OCR candidate.

    The score is only used to choose between simultaneously observed camera
    fields.  It never fabricates a level.  A single bad OCR digit therefore
    loses to a second field that remains monotonic and locally linear.
    """
    previous = series.shift(1)
    following = series.shift(-1)
    score = (series - (previous + following) / 2.0).abs()

    previous_step = series - previous
    following_step = following - series
    implausible = (
        previous_step.lt(-0.005)
        | following_step.lt(-0.005)
        | previous_step.gt(0.6)
        | following_step.gt(0.6)
    )
    score = score.where(~implausible, score.fillna(0.0) + 10.0)

    # At a source-boundary only one neighbour may exist.  Keep the same
    # physical monotonic/rate checks without pretending that a centred score
    # is available.
    one_sided = score.isna() & series.notna()
    one_step = previous_step.where(previous.notna(), following_step)
    score.loc[one_sided] = one_step.loc[one_sided].abs()
    score.loc[one_sided & (one_step.lt(-0.005) | one_step.gt(0.6))] = 10.0
    return score


def robust_cumulative(df: pd.DataFrame, camera: str) -> pd.DataFrame:
    """Fuse total and HFO+MDO cumulative OCR fields with auditable QC.

    Older preprocessing took the median of two values.  With two candidates
    that is their arithmetic mean, so one erroneous field could create an
    artificial low/high increment pair.  The revised rule first estimates the
    camera-specific total-minus-parts offset, accepts agreeing candidates, and
    resolves disagreements by local temporal consistency.  Ambiguous rows are
    kept missing instead of silently manufacturing ground truth.
    """
    total = df[f"{camera}_total"].where(df[f"{camera}_total"].between(10_000, 100_000))
    parts_raw = df[f"{camera}_hfo"] + df[f"{camera}_mdo"]
    parts_raw = parts_raw.where(parts_raw.between(10_000, 100_000))

    raw_offset = (total - parts_raw).where(total.notna() & parts_raw.notna())
    local_offset = raw_offset.rolling(73, center=True, min_periods=6).median()
    global_offset = float(raw_offset.median()) if raw_offset.notna().any() else 0.0
    local_offset = local_offset.fillna(global_offset)
    parts = parts_raw + local_offset

    total_score = _candidate_temporal_score(total)
    parts_score = _candidate_temporal_score(parts)
    residual = total - parts
    agree = total.notna() & parts.notna() & residual.abs().le(0.035)

    level = pd.Series(np.nan, index=df.index, dtype=float)
    source = pd.Series("missing", index=df.index, dtype=object)
    flag = pd.Series("missing_both", index=df.index, dtype=object)

    level.loc[agree] = (total.loc[agree] + parts.loc[agree]) / 2.0
    source.loc[agree] = "total_parts_consensus"
    flag.loc[agree] = "ok"

    disagreement = total.notna() & parts.notna() & ~agree
    choose_total = disagreement & (total_score + 0.02 < parts_score)
    choose_parts = disagreement & (parts_score + 0.02 < total_score)
    level.loc[choose_total] = total.loc[choose_total]
    source.loc[choose_total] = "total_temporal"
    flag.loc[choose_total] = "cross_field_disagreement"
    level.loc[choose_parts] = parts.loc[choose_parts]
    source.loc[choose_parts] = "parts_temporal"
    flag.loc[choose_parts] = "cross_field_disagreement"

    only_total = total.notna() & parts.isna()
    only_parts = parts.notna() & total.isna()
    total_ok = only_total & total_score.lt(1.0)
    parts_ok = only_parts & parts_score.lt(1.0)
    level.loc[total_ok] = total.loc[total_ok]
    source.loc[total_ok] = "total_only"
    flag.loc[total_ok] = "single_field"
    level.loc[parts_ok] = parts.loc[parts_ok]
    source.loc[parts_ok] = "parts_only"
    flag.loc[parts_ok] = "single_field"

    unresolved = disagreement & level.isna()
    flag.loc[unresolved] = "ambiguous_disagreement"
    rejected_single = (only_total | only_parts) & level.isna()
    flag.loc[rejected_single] = "implausible_single_field"

    return pd.DataFrame(
        {
            "level_raw_qc": level,
            "qc_source": source,
            "qc_flag": flag,
            "cross_field_residual": residual,
            "total_candidate": total,
            "parts_candidate": parts,
        },
        index=df.index,
    )


def interpolate_short(series: pd.Series, limit: int = 2, allowed: pd.Series | None = None) -> pd.Series:
    """Interpolate complete missing runs of at most ``limit`` grid steps.

    Pandas' ``limit`` fills the first rows of a much longer outage.  That made
    the beginning of long camera blackouts look like observed 10-minute GT.
    This implementation fills a run only when both bounding observations
    exist and the entire missing run is short.
    """
    out = series.copy()
    missing = out.isna()
    groups = (missing != missing.shift()).cumsum()
    for _, block in out.groupby(groups):
        if not block.isna().all() or len(block) > limit:
            continue
        if allowed is not None and not allowed.reindex(block.index).fillna(False).all():
            continue
        start_pos = out.index.get_loc(block.index[0])
        end_pos = out.index.get_loc(block.index[-1])
        if start_pos == 0 or end_pos == len(out) - 1:
            continue
        left, right = out.iloc[start_pos - 1], out.iloc[end_pos + 1]
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        values = np.linspace(float(left), float(right), len(block) + 2)[1:-1]
        out.loc[block.index] = values
    return out


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    idx = frame.index
    h = idx.hour + idx.minute / 60.0
    dow = idx.dayofweek
    frame["hour_sin"] = np.sin(2 * np.pi * h / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * h / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    frame["wind_vec_speed"] = np.hypot(frame["wx"], frame["wy"])
    frame["current_speed"] = np.hypot(frame["curx"], frame["cury"])
    encounter = np.deg2rad((frame["dirpw"] - frame["HEADING"]) % 360)
    frame["wave_encounter_cos"] = np.cos(encounter)
    frame["wave_encounter_sin"] = np.sin(encounter)
    frame["mean_rpm"] = frame[["PORT_RPM_OUTPUT", "STBD_RPM_OUTPUT"]].mean(axis=1)
    frame["mean_pitch"] = frame[["PORT_PITCH_OUTPUT", "STBD_PITCH_OUTPUT"]].mean(axis=1)
    engine_status = frame[["PORT_ENGINE_STATUS", "STBD_ENGINE_STATUS"]]
    frame["engine_on"] = (engine_status.max(axis=1) >= 2).astype(float).where(engine_status.notna().any(axis=1))
    return frame


def prepare(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(csv_path, low_memory=False)
    dt = pd.to_datetime(raw.pop("DateTime"), errors="raise")
    raw.index = dt
    raw = raw[~raw.index.duplicated(keep="first")].sort_index()
    numeric = numeric_frame(raw)

    grid = pd.date_range(raw.index.min().floor("10min"), raw.index.max().ceil("10min"), freq="10min")
    frame = numeric.reindex(grid)
    frame.index.name = "timestamp"
    sensor_columns = [column for column in BASE_FEATURES if column in frame]
    sensors_present = frame[sensor_columns].notna().any(axis=1)
    vessel_columns = [column for column in BASE_FEATURES[:12] if column in frame]
    frame["vessel_observed"] = frame[vessel_columns].notna().any(axis=1).astype(float)
    frame["sensor_observed"] = sensors_present.astype(float)
    frame["gap_type"] = np.where(sensors_present, "sensors_present", "suspected_docked_all_sensors_missing")
    for camera in ("cam1", "cam2"):
        qc = robust_cumulative(frame, camera)
        raw_level = qc["level_raw_qc"]
        frame[f"{camera}_level_clean"] = interpolate_short(raw_level, limit=2, allowed=sensors_present)
        frame[f"{camera}_level_observed"] = raw_level.notna().astype(float)
        frame[f"{camera}_qc_source"] = qc["qc_source"]
        frame[f"{camera}_qc_flag"] = qc["qc_flag"]
        frame[f"{camera}_cross_field_residual"] = qc["cross_field_residual"]
        camera_delta = frame[f"{camera}_level_clean"].diff()
        frame[f"{camera}_step"] = camera_delta.where(camera_delta.between(0.0, 0.6))

    frame.loc[sensors_present & frame.cam2_level_observed.eq(0), "gap_type"] = "fuel_missing_sensors_present"

    frame["fuel_level"] = frame["cam1_level_clean"] + frame["cam2_level_clean"]
    delta = frame["fuel_level"].diff()
    # A 10-minute increment above 1 MT or a negative increment is an OCR/path error.
    frame["fuel_step"] = delta.where(delta.between(0.0, 1.0))
    frame = add_features(frame)

    midnight = frame.loc[frame.index.time == pd.Timestamp("00:00").time(), [
        "cam1_level_clean", "cam2_level_clean", "fuel_level"
    ]].copy()
    daily = midnight.diff()
    daily.columns = ["cam1_consumption", "cam2_consumption", "fuel_consumption"]
    # Difference at midnight t is consumption over [t-1 day, t); label by interval start.
    daily["date"] = daily.index.normalize() - pd.Timedelta(days=1)
    daily["observed"] = daily[["cam1_consumption", "cam2_consumption"]].notna().all(axis=1)
    daily["cam1_plausible"] = daily["cam1_consumption"].between(0, 50)
    daily["cam2_plausible"] = daily["cam2_consumption"].between(0, 50)
    daily["plausible"] = (
        daily["observed"]
        & daily["cam1_plausible"]
        & daily["cam2_plausible"]
        & daily["fuel_consumption"].between(0, 100)
    )

    day_key = frame.index.normalize()
    step_daily_sum = frame["fuel_step"].groupby(day_key).sum(min_count=120)
    step_coverage = frame["fuel_step"].notna().groupby(day_key).mean()
    feature_coverage = frame[BASE_FEATURES].notna().mean(axis=1).groupby(day_key).mean()
    daily["step_sum"] = daily["date"].map(step_daily_sum)
    daily["step_coverage"] = daily["date"].map(step_coverage)
    daily["feature_coverage"] = daily["date"].map(feature_coverage)

    valid = daily.loc[daily["plausible"], "fuel_consumption"]
    stats = {
        "rows_raw": int(len(raw)),
        "rows_grid": int(len(frame)),
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "days_total": int(len(daily)),
        "days_with_plausible_midnight_label": int(daily["plausible"].sum()),
        "cam1_days_with_plausible_label": int(daily["cam1_plausible"].sum()),
        "cam2_days_with_plausible_label": int(daily["cam2_plausible"].sum()),
        "fuel_daily_median": float(valid.median()),
        "fuel_daily_mean": float(valid.mean()),
        "fuel_daily_p01": float(valid.quantile(0.01)),
        "fuel_daily_p99": float(valid.quantile(0.99)),
        "median_step_coverage": float(daily["step_coverage"].median()),
        "median_feature_coverage": float(daily["feature_coverage"].median()),
        "midnight_vs_step_mae": float(
            (daily.loc[daily["plausible"], "fuel_consumption"] - daily.loc[daily["plausible"], "step_sum"])
            .abs().dropna().mean()
        ),
    }
    return frame, daily, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frame, daily, stats = prepare(args.csv)
    frame.to_csv(args.out / "kriso_10min.csv.gz", compression="gzip")
    daily.to_csv(args.out / "kriso_daily_labels.csv", index=False)
    (args.out / "preprocess_summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
