"""Sequential isolated raw/cache/base/final chain. Existing outputs fail closed."""
import os,sys,subprocess,time,json
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH='/workspace/condition_anchor_code_20261019/src')
R=Path('/workspace');O=R/'submission_raw_20261026';code=R/'condition_anchor_code_20261019'
start=time.monotonic()
def run(args):
    if time.monotonic()-start>21600:raise TimeoutError('overall six-hour compute gate')
    subprocess.run([sys.executable,*map(str,args)],check=True,timeout=max(1,21600-(time.monotonic()-start)))
assert json.loads((O/'raw_rebuild_audit.json').read_text())['complete']
run([R/'build_rolling_submission_20261026.py','--frame',O/'operational_qc_v3_regenerated_20261013/merged_qc_frame.csv.gz','--out',O/'fresh_fm_cache_20261013'])
run([code/'tools/train_cached_grpo_base.py','--root',O,'--out',O/'results_operational_fresh_fm_20261013_strict_full','--historical-selection'])
run([R/'submission_selected_retrain_20261026.py'])
print('RAW_TO_FINAL_COMPLETE',flush=True)
