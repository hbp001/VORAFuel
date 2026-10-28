"""Public inference interface to the unchanged, evaluated VORAFuel modules."""
from pathlib import Path
import sys
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / 'research' / 'src'))
from operational_core import Policy
from relational_ar_20261016 import SixHourPolicy, ARCHS
from plan_response_20261019 import PlanResponsePolicy, VARIANTS


class VORAFuel(nn.Module):
    """One single-seed predictor; no future fuel or ensemble input.

    history: B x 36 x 14 x 3 (newest first; normalized value, mask, age)
    fm: B x F (normalized frozen dual-stream cache plus presence and age)
    plan: B x 12 x 9 (see docs/INPUTS.md)
    """
    def __init__(self, fm_features=1539):
        super().__init__()
        self.fm_features = fm_features
        self.policy = PlanResponsePolicy(
            SixHourPolicy(Policy(288, 216, fm_features), ARCHS[2]), VARIANTS[0])

    def forward(self, history, fm, plan, source_flag=None):
        b = history.shape[0]
        expected = [(history, (b,36,14,3)), (fm, (b,self.fm_features)), (plan, (b,12,9))]
        for tensor, shape in expected:
            if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
                raise ValueError(f'Expected finite tensor with shape {shape}')
        if source_flag is None:
            source_flag = history.new_zeros(b,1)
        if tuple(source_flag.shape) != (b,1) or not torch.isfinite(source_flag).all():
            raise ValueError('source_flag must be finite B x 1')
        context, memory = self.policy.backbone.encode(history, fm)
        # The selected variant has no raw-condition response branch.
        raw = history.new_zeros(b,14)
        return .3 * self.policy.predict_encoded((context,memory,raw,source_flag), plan)

    @classmethod
    def from_checkpoint(cls, path):
        """Load trusted author-provided tensor checkpoint, on CPU."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint.get('state', checkpoint)
        model = cls(state['backbone.fm.0.weight'].shape[1])
        model.policy.load_state_dict(state, strict=True)
        return model.eval()


if __name__ == '__main__':
    import argparse
    import numpy as np
    p=argparse.ArgumentParser(description='Inference from preprocessed tensors; no fitting.')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--inputs',required=True,help='NPZ with history, fm, plan; optional source_flag')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if Path(a.output).exists():
        p.error('Output exists; choose a new path')
    torch.set_num_threads(4)
    m=VORAFuel.from_checkpoint(a.checkpoint)
    with np.load(a.inputs,allow_pickle=False) as z, torch.inference_mode():
        args=[torch.tensor(z[k],dtype=torch.float32) for k in ['history','fm','plan']]
        flag=torch.tensor(z['source_flag'],dtype=torch.float32) if 'source_flag' in z else None
        y=m(*args,source_flag=flag).numpy()
    np.savez_compressed(a.output,prediction=y,total=y.sum(axis=1))
