# GPU execution verification

Verified on 2026-10-28 with NVIDIA RTX A6000 and PyTorch 2.7.1+cu126.

`python -m unittest discover -s tests -v` passed all five tests in an isolated
CUDA container. The CUDA test loads a synthetic checkpoint, compares CPU and
GPU predictions (`rtol=2e-4`, `atol=2e-5`), and checks finite backward gradients.
It performs no optimizer update. This verifies device compatibility; it is not
a new GPU benchmark or a re-evaluation of manuscript accuracy.

The saved-prediction reproduction checks in `VERIFICATION.json` remain the
CPU reference. Model weights and research datasets are not included here.
