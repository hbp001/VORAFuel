#!/usr/bin/env python3
"""Preserve existing fuel data; append collected operational KMA context."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES=['kma_fc_wave_m','kma_fc_period_s','kma_fc_wind_u','kma_fc_wind_v','kma_fc_wave_dir_sin','kma_fc_wave_dir_cos']

def main():
    p=argparse.ArgumentParser();p.add_argument('--frame',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--forecasts',type=Path,required=True);p.add_argument('--extraction-manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    base=pd.read_csv(a.frame,parse_dates=['timestamp']); original=base.copy(deep=True)
    fc=pd.read_csv(a.forecasts)
    fc['valid_key']=fc.valid_utc.astype(str)
    fc['available_utc']=pd.to_datetime(fc.analysis_utc.astype(str),format='%Y%m%d%H',utc=True).astype('datetime64[ns, UTC]')+pd.Timedelta(hours=6)
    fc['valid_utc_parsed']=pd.to_datetime(fc.valid_key,format='%Y%m%d%H',utc=True).astype('datetime64[ns, UTC]')
    assert (fc.valid_utc_parsed>=fc.available_utc-pd.Timedelta(hours=6)).all()
    zones=fc[['zone','lat','lon']].drop_duplicates('zone').set_index('zone')
    centers=np.c_[zones.lat*111,zones.lon*89]
    geo=base.LAT.between(35.2,38.2)&base.LON.between(121.7,127.4)
    left=pd.DataFrame({'row':np.flatnonzero(geo)})
    coordinates=base.loc[geo,['LAT','LON']].to_numpy()*[111,89]
    selections=[];distances=[]
    for start in range(0,len(coordinates),1024):
        d=np.sqrt(((coordinates[start:start+1024,None,:]-centers[None,:,:])**2).sum(axis=2))
        j=d.argmin(axis=1);selections.extend(j);distances.extend(d[np.arange(len(j)),j])
    index=np.asarray(selections);distance=np.asarray(distances)
    left['zone']=zones.index.to_numpy()[index];left['distance_km']=distance
    left['decision_utc']=pd.to_datetime(base.loc[geo,'timestamp']).dt.tz_localize('Asia/Seoul').dt.tz_convert('UTC').to_numpy()
    left['decision_utc']=pd.to_datetime(left.decision_utc,utc=True).astype('datetime64[ns, UTC]')
    left['valid_key']=left.decision_utc.dt.floor('3h').dt.strftime('%Y%m%d%H')
    joined=pd.merge_asof(left.sort_values('decision_utc'),fc.sort_values('available_utc'),left_on='decision_utc',right_on='available_utc',by=['zone','valid_key'],allow_exact_matches=True)
    ok=joined.available_utc.notna()&(joined.distance_km<=50)
    present=joined.loc[ok]
    assert (present.available_utc<=present.decision_utc).all()
    assert (present.valid_utc_parsed<=present.decision_utc).all()
    for field,(lo,hi) in {'wave_m':(0,20),'period_s':(0,40),'wind_ms':(0,100),'wave_direction_deg':(0,360),'wind_direction_deg':(0,360)}.items():
        joined[field]=joined[field].where(joined[field].between(lo,hi)&ok)
    # Meteorological wind direction is FROM: convert to east/north flow.
    joined['kma_fc_wave_m']=joined.wave_m; joined['kma_fc_period_s']=joined.period_s
    rad=np.deg2rad(joined.wind_direction_deg)
    joined['kma_fc_wind_u']=-joined.wind_ms*np.sin(rad);joined['kma_fc_wind_v']=-joined.wind_ms*np.cos(rad)
    rad=np.deg2rad(joined.wave_direction_deg)
    joined['kma_fc_wave_dir_sin']=np.sin(rad);joined['kma_fc_wave_dir_cos']=np.cos(rad)
    for c in FEATURES:
        base[c]=np.nan;base.loc[joined.row,c]=joined[c].to_numpy()
    base['kma_fc_available']=base.kma_fc_wave_m.notna().astype(int)
    pd.testing.assert_frame_equal(base[original.columns],original)
    output=a.out/'kriso_collected_10min.csv.gz';base.to_csv(output,index=False,compression='gzip')
    provenance=joined[['row','decision_utc','available_utc','valid_utc_parsed','zone','distance_km','source_file']].copy()
    provenance['join_accepted']=ok.to_numpy();provenance.to_csv(a.out/'kma_join_provenance.csv.gz',index=False,compression='gzip')
    shutil.copy2(a.labels,a.out/'kriso_daily_labels.csv')
    labels=pd.read_csv(a.labels,parse_dates=['date']);good=labels.cam2_plausible.fillna(False)
    split=labels.loc[good,['date']].copy();split['split']=np.where(split.date<'2026-09-11','train',np.where(split.date<'2026-09-21','validation','exploratory_test'))
    split.to_csv(a.out/'split_manifest.csv',index=False)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    summary=dict(rows=len(base),daily_plausible_labels=int(good.sum()),split_days=split.split.value_counts().to_dict(),feature_columns=FEATURES,feature_available_rows={c:int(base[c].notna().sum()) for c in FEATURES},existing_columns_unchanged=True,labels_sha256=sha(a.labels),base_sha256=sha(a.frame),output_sha256=sha(output),kma_snapshot=json.loads(a.extraction_manifest.read_text()),timezone='ship naive timestamps assumed KST; forecast UTC',alignment='nearest zone center <=50km, latest cycle with analysis+6h<=input timestamp; valid_time=floor(inputUTC,3h)',publication_delay_hours_assumed=6,not_verified_provider_issue_time=True,scope='offline collected-only KRISO downstream; existing ERA5/CMEMS retained hindcast context, not fully operational forecast evaluation',excluded=['UM/KIM meteorological data not collected','besttrack unavailable','latest sea forecast outside fuel period','typhoon retrospective inputs','observation stations pending physical/QC join','partial small-zone archive'],training_ready=True,trained=False,new_holdout_required_for_confirmation=True)
    (a.out/'training_ready_manifest.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in summary.items() if k!='kma_snapshot'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
