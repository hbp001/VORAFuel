# VORAFuel

**VORAFuel: Ship Fuel Consumption Prediction with Observation-Availability-Aware Relational Memory and Operationally Conditioned Autoregressive Learning**

Maintained by **Hyobin Park** ([hbp001](https://github.com/hbp001)).

Code accompanying the Korean manuscript in preparation. No journal acceptance,
DOI, or publication date is claimed.

VORAFuel combines observation values, masks and ages, alternating temporal and
variable attention, frozen time-series foundation-model context, and a single
conditioned autoregressive fuel decoder. The public interface uses the evaluated
modules without changing their equations or parameter names.

## Quick start

Python 3.11, CPU:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The tests use generated tensors, download no models or data, and do not reproduce
the manuscript scores. With approved private weights and preprocessed inputs:

```bash
python vorafuel.py --checkpoint /private/final.pt \
  --inputs /private/inputs.npz --output /new/location/prediction.npz
```

## Method and implementation

```text
Vessel history ─┬─ value / mask / age ─ temporal + variable attention ─ memory ─┐
Weather history┘                                                            │
Past-only frozen Chronos-2 cache ─ projected global context ──────────────────┤
Available speed/heading conditions ───────────────────────── single AR decoder
                                                               │
                                               fuel trajectory + summed demand
```

The evaluated implementation uses 36 ten-minute history steps and 12 half-hour
output bins. These are experimental settings, not a claim that arbitrary horizon
changes retain accuracy. Fixed-size layers in this release require adaptation and
validation for other settings. Future observed fuel is never an inference input.
Real future vessel motion is an oracle diagnostic, distinct from available plans.

- [Input contract](docs/INPUTS.md)
- [Reproduction and training lineage](docs/REPRODUCIBILITY.md)
- [Evidence and scope](docs/RESULTS.md)
- [Data and weights requests](docs/ACCESS.md)

## Repository contents

`vorafuel.py` is the clean inference entry point. `research/src` and
`research/tools` preserve the evaluated training, preprocessing and comparison
code; dated names retain traceability. `SOURCE_MANIFEST.json` records their hashes.
Historical alternative branches are research controls, not additional components
of the selected model. No raw/processed measurements, timestamps with GT,
trajectories, private checkpoints, credentials or Git development history are
included. Third-party FM weights must be obtained under their upstream terms.

## Availability and rights

The dataset and task-specific model weights remain **private** and are available
**on request**, subject to applicable provider permissions. Open a repository issue to initiate contact; do not post
private data or credentials in the issue. See [access terms](docs/ACCESS.md).
This code release does not grant rights to KRISO or weather-provider data.
A code license has not yet been designated by the authors; this is a public source
release, not a claim of an OSI-approved open-source license.
