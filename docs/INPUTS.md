# Input and output contract

All tensors are float32 on the same device. Never fill an unobserved fuel target
with zero. A zero in a normalized input is meaningful only together with its mask
and age; it does not certify an observed zero consumption.

## History: B × 36 × 14 × 3

Newest first. Three channels: normalized value, fresh observation mask, age.
The archived preprocessing applies a three-native-step reception delay (30 min),
bounded past holding, and training-only normalization. Use `causal_channels` in
`research/src/operational_core.py` and `histories` in
`research/src/relational_ar_20261016.py`; do not normalize per inference window.

Variables, in order:

1. SOG
2. mean_rpm
3. mean_pitch
4. PORT_ENGINE_STATUS
5. STBD_ENGINE_STATUS
6. heading_sin
7. heading_cos
8. raw_fuel (past interval increment)
9. kma_fc_wave_m
10. kma_fc_period_s
11. kma_fc_wind_u
12. kma_fc_wind_v
13. kma_fc_wave_dir_sin
14. kma_fc_wave_dir_cos

Position is used by environmental preprocessing; it is not an extra direct
decoder variable. Do not silently append vessel speed-through-water or new UM
temperature channels to this checkpoint.

## FM: B × 1539

Two frozen Chronos-2 embeddings (768 each), normalized using training statistics,
followed by two group-presence flags and cache age. Use the archived `features`
function in `research/src/replay_grpo_base_20261015.py`. The cache is past-only;
it is refreshed hourly from 12 native historical steps per group.

## Plan: B × 12 × 9

Each future output interval has: speed/50, heading sine, heading cosine,
speed-validity flag, heading-validity flag, lead/144, sine and cosine of the
target clock hour, interval-duration/144. Lead and duration are in ten-minute
steps. Missing speed/heading slots are zero with validity flags zero.

Historical evaluation uses a preceding-received-day synthetic plan, not actual
future motion. `plan_tokens(..., MODES[1])[:,1:13]` implements it. An operational
plan must have been available at prediction time. Source flag is zero for the
historical synthetic condition and one for the supplied-condition diagnostic;
the latter is not evidence of validated real-plan operation.

## Output

`prediction`: B × 12 nonnegative interval amounts; `total`: sum over 12 bins.
The internal .3 scaling reproduces the archived implementation. MT/30min is the
manuscript's provisional interpretation pending KRISO meter confirmation.
One call generates a whole future sequence without future fuel GT. No online
weight updates occur in the public inference entry point.
