"""Rebuild the final model's used data path from immutable provider CSVs and KMA bytes.
Unused ERA5/CMEMS/KHOA research columns are deliberately outside this dependency path.
No network, credentials, target imputation, or replacement of archived artifacts.
"""
import json,sys,subprocess,hashlib
from pathlib import Path
import pandas as pd
import numpy as np
R=Path('/workspace'); O=R/'submission_raw_20261026'
sys.path[:0]=[str(R/'condition_anchor_code_20261019/src'),str(R/'src')]
from prepare_kriso import prepare
from operational_core import causal_channels,raw_target,WEATHER
from short_gap_labels import NATIVE
from relational_ar_20261016 import six_hour_targets
from planned_multiscale_20261015 import plan_tokens,MODES
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    oldpath=O/'old.csv';newpath=O/'new.csv'
    assert sha(oldpath)=='ba01cfed601e37f207b10bf95bc92ca7697b48bbfc12d6261b45c0486e8245dc'
    assert sha(newpath)=='d70e1e193fa3d552576dd20a74829c039e0a2fb1b32bef76f2e65a807aec1004'
    old=pd.read_csv(oldpath,low_memory=False).set_index('DateTime');new=pd.read_csv(newpath,low_memory=False).set_index('DateTime')
    old.index=pd.to_datetime(old.index);new.index=pd.to_datetime(new.index)
    old=old.apply(pd.to_numeric,errors='coerce');n=new.reindex(columns=old.columns).apply(pd.to_numeric,errors='coerce')
    for camera in ['cam1','cam2']:
        n.loc[~new[camera+'_status'].eq('complete'),[c for c in n if c.startswith(camera+'_')]]=np.nan
    merged=old.combine_first(n).sort_index();merged.index.name='DateTime';merged.to_csv(O/'merged_raw.csv.gz')
    frame,daily,stats=prepare(O/'merged_raw.csv.gz')
    frame.to_csv(O/'ship_10min.csv.gz');daily.to_csv(O/'daily.csv',index=False)
    subprocess.run([sys.executable,str(R/'prepare_collected_submission_20261026.py'),'--frame',str(O/'ship_10min.csv.gz'),'--labels',str(O/'daily.csv'),'--forecasts',str(O/'kma/kma_large_zone_bbox.csv.gz'),'--extraction-manifest',str(O/'kma/extraction_manifest.json'),'--out',str(O/'joined')],check=True)
    rebuilt=pd.read_csv(O/'joined/kriso_collected_10min.csv.gz',parse_dates=['timestamp']).set_index('timestamp')
    archived=pd.read_csv(R/'operational_qc_v3_regenerated_20261013/merged_qc_frame.csv.gz',parse_dates=['timestamp']).set_index('timestamp')
    pd.testing.assert_index_equal(rebuilt.index,archived.index)
    cols=NATIVE+['cam2_total','cam2_hfo','cam2_mdo','mean_rpm','mean_pitch']+WEATHER
    checks={}
    for c in cols:
        np.testing.assert_allclose(rebuilt[c],archived[c],rtol=0,atol=1e-10,equal_nan=True)
        checks[c]=int(rebuilt[c].notna().sum())
    for fresh,prior in zip(causal_channels(rebuilt,3),causal_channels(archived,3)):
        for a,b in zip(fresh,prior):np.testing.assert_allclose(a,b,rtol=0,atol=1e-10,equal_nan=True)
    for a,b in zip(six_hour_targets(rebuilt),six_hour_targets(archived)):np.testing.assert_allclose(a,b,rtol=0,atol=1e-10,equal_nan=True)
    ids=np.arange(36,len(rebuilt),2)
    np.testing.assert_array_equal(plan_tokens(rebuilt,ids,MODES[1]),plan_tokens(archived,ids,MODES[1]))
    dest=O/'operational_qc_v3_regenerated_20261013';dest.mkdir()
    rebuilt.to_csv(dest/'merged_qc_frame.csv.gz')
    report=dict(complete=True,rows=len(rebuilt),old_sha256=sha(oldpath),new_sha256=sha(newpath),output_sha256=sha(dest/'merged_qc_frame.csv.gz'),checked_columns=checks,causal_values_mask_age_equal=True,targets_release_equal=True,causal_plan_equal=True,tolerance=1e-10,scope='all raw fields used by final model; unused retrospective ERA5/CMEMS/KHOA excluded',kma=json.loads((O/'kma/extraction_manifest.json').read_text()))
    (O/'raw_rebuild_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print('RAW_PATH_VERIFIED',flush=True)
if __name__=='__main__':main()
