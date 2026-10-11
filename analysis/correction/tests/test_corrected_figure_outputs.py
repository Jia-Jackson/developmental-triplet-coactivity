from pathlib import Path
import numpy as np
import pandas as pd
from hotopology.story_contract import p_to_symbol

ROOT=Path(__file__).resolve().parents[1]
T=ROOT/'tables/computational_closure_v2'

def test_current_six_main_ten_supplement_assemblies():
    labels=[str(i) for i in range(1,7)]+[f'S{i}' for i in range(1,11)]
    for label in labels:
        for ext in ('png','pdf','tiff'):
            assert (ROOT/f'figures/assembled/Figure_{label}.{ext}').stat().st_size>1000

def test_current_numerical_panels_and_statistics_use_valid_symbols():
    files=list((ROOT/'figures/individual').rglob('*_statistics.csv'))
    assert len(files)>=20
    for path in files:
        table=pd.read_csv(path)
        if {'symbol','adjusted_p'} <= set(table):
            finite=table.loc[np.isfinite(table.adjusted_p)]
            assert finite.symbol.tolist()==finite.adjusted_p.map(p_to_symbol).tolist(),str(path)

def test_current_hierarchy_uses_exactly_the_null100_rows():
    from run_v2_hierarchical_statistics import _assemble_datasets
    actual=_assemble_datasets(ROOT)['conditioned_triplet_residual'][0].set_index('recording').sort_index()
    expected=pd.read_csv(T/'v2_instantaneous_null100_triplet_residuals.csv').set_index('recording').sort_index()
    assert len(actual)==len(expected)==142
    assert actual.index.equals(expected.index)
    np.testing.assert_allclose(actual.absolute_residual_z,expected.absolute_residual_z,rtol=1e-12)

def test_endpoint_missing_does_not_drop_other_outcomes():
    table=pd.read_csv(T/'v2_kinetic_null_comparisons.csv')
    n=table.groupby('metric').z.count()
    assert n['h1_infinite_count']==141
    assert n['h1_total_persistence_capped']==142
    missing=table.loc[table.z.isna()]
    assert len(missing)==1 and missing.iloc[0].recording=='WT_DIV63_P3_A1'
    assert missing.iloc[0].observed==1 and missing.iloc[0].null_mean==0
    stats=pd.read_csv(T/'v2_null_deviation_one_sample_statistics.csv')
    row=stats.loc[stats.model.eq('kinetic_conditioned_k')&stats.source_metric.eq('h1_infinite_count')&stats.DIV.eq(63)]
    assert len(row)==1 and row.iloc[0]['n']==16
