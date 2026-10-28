"""Repeat frozen classical/FM baseline protocol on the rebuilt final-model data."""
import os,sys,json
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
R=Path('/workspace');O=R/'submission_raw_20261026'
sys.path[:0]=[str(R/'condition_anchor_code_20261019/tools'),str(R/'condition_anchor_code_20261019/src')]
import numpy as np
import run_plan_response_20261019 as p
import run_unified_baselines_20261019 as b
from replay_grpo_base_20261015 import sha,BASE
p.SOURCE_SHA=sha(O/'operational_qc_v3_regenerated_20261013/merged_qc_frame.csv.gz');p.CACHE_SHA=sha(O/'fresh_fm_cache_20261013/fresh_dual_cache.npz');p.SCALER_SHA=sha(O/BASE/'scalers.pt')
folder=O/'plan_response_full_v2_20261019';folder.mkdir()
(folder/'protocol.json').write_text(json.dumps(dict(source_sha256=p.SOURCE_SHA)))
for phase in ['classical','foundation']:
    out=O/('unified_'+phase);sys.argv=['baseline','--root',str(O),'--out',str(out),'--phase',phase];b.main()
    comparisons=[]
    for path in out.glob('*.npz'):
        prior=R/('unified_'+phase+'_20261019')/path.name
        assert prior.exists();a=np.load(path);z=np.load(prior)
        np.testing.assert_array_equal(a['issues'],z['issues']);np.testing.assert_allclose(a['target'],z['target'],rtol=0,atol=0,equal_nan=True)
        comparisons.append(dict(file=path.name,maximum_prediction_difference=float(np.max(np.abs(a['prediction']-z['prediction'])))))
    (out/'reproduction_comparison.json').write_text(json.dumps(comparisons,indent=2))
print('BASELINES_REPEATED',flush=True)
