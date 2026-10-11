"""Correct only the approved undefined endpoint score, preserving audit copies."""
from pathlib import Path
import hashlib, json, shutil
import pandas as pd
from hotopology.endpoint_validity import mark_undefined_endpoint_z

ROOT=Path(__file__).resolve().parents[1]
T=ROOT/'tables/computational_closure_v2'

def main():
    archive=ROOT/'audit/pre_undefined_endpoint'; archive.mkdir(parents=True,exist_ok=True)
    filenames=['v2_kinetic_null_comparisons.csv','v2_extended_null_hierarchical_coefficients.csv','v2_extended_null_hierarchical_contrasts.csv','v2_extended_null_culture_changes.csv','v2_null_deviation_one_sample_statistics.csv','v2_instantaneous_vs_kinetic_paired_statistics.csv']
    for name in filenames:
        if not (archive/name).exists(): shutil.copy2(T/name,archive/name)
    x=pd.read_csv(T/filenames[0]); y=mark_undefined_endpoint_z(x)
    bad=y.loc[~y.z_available]
    if len(bad)!=1 or bad.iloc[0].recording!='WT_DIV63_P3_A1' or bad.iloc[0].metric!='h1_infinite_count':
        raise RuntimeError('Unexpected undefined-score scope')
    y.to_csv(T/filenames[0],index=False)
    bad.to_csv(T/'undefined_endpoint_score_audit.csv',index=False)
    counts=y.groupby(['model','metric','DIV']).agg(n_recordings=('recording','size'),n_available_z=('z','count')).reset_index()
    counts.to_csv(T/'kinetic_metric_sample_sizes.csv',index=False)
    report={'approved_policy':'Only undefined endpoint z is missing; observations, simulations and empirical P retained','record':bad.to_dict('records'),'n_kinetic_records':y.recording.nunique(),'n_endpoint_z':int(y.loc[y.metric.eq('h1_infinite_count'),'z'].count()),'preserved_checkpoint':'Original numeric sentinel retained in raw checkpoint as audit; export adapter always converts it to missing','old_table_sha256':hashlib.sha256((archive/filenames[0]).read_bytes()).hexdigest(),'new_table_sha256':hashlib.sha256((T/filenames[0]).read_bytes()).hexdigest()}
    (ROOT/'logs/undefined_endpoint_correction.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
