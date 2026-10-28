"""Build a label-independent CPU rolling Chronos-2 cache from private inputs."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from operational_core import causal_channels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frame', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', default='amazon/chronos-2')
    p.add_argument('--allow-model-download', action='store_true')
    a = p.parse_args()
    from chronos import Chronos2Pipeline
    a.out.mkdir(parents=True, exist_ok=True)
    lock = (a.out/'worker.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (a.out/'manifest.json').exists():
        raise FileExistsError('Completed cache exists; validate it rather than rebuilding')
    source_sha = hashlib.sha256(a.frame.read_bytes()).hexdigest()
    frame = pd.read_csv(a.frame, parse_dates=['timestamp']).set_index('timestamp')
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    assert frame.index.to_series().diff().dropna().eq(pd.Timedelta(minutes=10)).all()
    blocks = causal_channels(frame, 3); indices = np.arange(14, len(frame), 6)
    torch.set_num_threads(4); began = time.monotonic()
    pipe = Chronos2Pipeline.from_pretrained(a.model, device_map='cpu', dtype=torch.float32,
                                          local_files_only=not a.allow_model_download)
    chunks = a.out/'chunks'; chunks.mkdir(exist_ok=True)
    vectors, flags = [], []
    for start in range(0, len(indices), 64):
        take = indices[start:start+64]; dest = chunks/f'{start:06d}.npz'
        if dest.exists():
            saved = np.load(dest, allow_pickle=False)
            assert str(saved['source_sha']) == source_sha and np.array_equal(saved['indices'], take)
            assert str(saved['model']) == a.model
            vector, present = saved['vectors'], saved['present']
        else:
            inputs, mask = [], []
            for i in take:
                for values, _, _ in blocks:
                    block = values[i-11:i+1].astype(np.float32).T.copy()
                    observed = np.isfinite(block).any(); mask.append(observed)
                    if not observed: block[0, -1] = 0.
                    inputs.append(block)
            with torch.inference_mode(): embedded, _ = pipe.embed(inputs, batch_size=32, context_length=12)
            values = np.stack([v.float().mean(dim=(0, 1)).numpy() for v in embedded])
            values[~np.array(mask)] = 0.
            vector = np.c_[values[::2], values[1::2]]; present = np.array(mask).reshape(-1, 2)
            temporary = dest.with_suffix('.part.npz')
            np.savez_compressed(temporary, vectors=vector, present=present, indices=take,
                                source_sha=source_sha, model=a.model)
            temporary.replace(dest)
        assert vector.shape == (len(take), 1536) and np.isfinite(vector).all()
        vectors.append(vector); flags.append(present)
    dest = a.out/'fresh_dual_cache.npz'
    np.savez_compressed(dest, x=np.concatenate(vectors), group_present=np.concatenate(flags),
                        issue_indices=indices, source_sha=source_sha)
    manifest = dict(complete=True, source_sha256=source_sha,
        cache_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), issues=len(indices),
        model=a.model, device='cpu', label_independent=True, context_steps=12,
        reception_delay_minutes=30, refresh_minutes=60, elapsed_seconds=time.monotonic()-began,
        scope='frozen raw dual-stream Chronos cache; not learned upstream encoder-to-FM')
    (a.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == '__main__': main()
