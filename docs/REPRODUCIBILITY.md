# Reproduction

## Two separate purposes

`vorafuel.py` loads one final checkpoint and predicts from prepared tensors.
`research/` preserves the historical training and comparison implementations.
The latter have fixed archive paths and require approved private artifacts. They
are not an automatically executable training job on checkout or installation.

## Selected training lineage

1. Raw counter QC, operational preprocessing and weather matching.
2. Past-only frozen Chronos-2 dual-stream cache and training statistics.
3. Cached base SFT (50 epochs) and offline GRPO initialization.
4. Relation encoder warm training and autoregressive decoder updates.
5. Total/low-load objective repair.
6. Selected causal-plan training with `VARIANTS[0]` (no optional response head).

Seeds 13, 37 and 71 are separate runs, never an ensemble. The final architecture
is `PlanResponsePolicy(SixHourPolicy(Policy(...), ARCHS[2]), VARIANTS[0])`.
The full selected sequence and optimizer counts are in
`research/tools/submission_selected_retrain_20261026.py`.
Do not run all historical candidate branches and label the best new result as
the frozen manuscript result.

## Archived environment and paths

The audited environment used Python 3.11, torch 2.7.1, NumPy 2.2.6, pandas 3.0.6,
CPU four threads, CUDA disabled and a 24 GiB memory limit. Optional packages for
baseline/cache reproduction are listed in the accompanying private environment
record; the lightweight inference requirements do not include all baselines.

Historical scripts expect `/workspace` as the approved private working root.
Place `research/` there as `condition_anchor_code_20261019/`; copy submission and
reviewer entry scripts from `research/tools/` to `/workspace/` where referenced.
Set `PYTHONPATH=/workspace/condition_anchor_code_20261019/src:/workspace/condition_anchor_code_20261019/tools`.
Keep private source files and reference checkpoints read-only. Use a separate
working copy for regeneration: completed output directories intentionally fail
closed rather than overwrite existing results. The raw replay requires provider
CSVs, extracted weather inputs and original reference artifacts for equality
checks, not merely the final weights.

`submission_raw_rebuild_20261026.py` reconstructs and checks the input path;
`submission_pipeline_20261026.py` runs cache/base/selected-model reconstruction;
`submission_baseline_rerun_20261026.py` handles frozen baseline protocols.
`submission_runtime_v3_20261026.py` is the matched timing audit. Earlier timing
scripts remain archival references, not the final timing result.

## Evaluation

Use identical output IDs, observed-target masks and seeds when comparing models.
Compute complete-window totals only where all target bins are observed; preserve
the missing-target mask. Randomly initialized 640-update controls and rolling
period controls are separate experiments from the multistage final model.
Day-block bootstrap and non-overlapping windows address dependence, but do not
turn development-reviewed dates into an independent holdout.

Source hashes are preserved in `SOURCE_MANIFEST.json`. Public-interface tests
check checkpoint restoration, equivalence to the archived decoder and causal
prefix behavior. Private saved-prediction agreement is recorded separately in
the release verification receipt. Do not load untrusted pickle-based scaler or
historical checkpoint files.
