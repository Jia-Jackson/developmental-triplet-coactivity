from pathlib import Path
import json
import pandas as pd
import pytest
from hotopology.current_null import load_instantaneous_null100


def write_fixture(root, drop=False):
    out = root / 'tables/computational_closure_v2'
    out.mkdir(parents=True)
    a, b = 'WT_DIV14_P1_A1', 'WT_DIV63_P3_B3'
    pd.DataFrame([dict(recording=a, analysis='dynamic16', eligible=True),
                  dict(recording=b, analysis='dynamic16', eligible=False)]).to_csv(
        out/'requested_node_eligibility.csv', index=False)
    base = dict(recording=a, model='pairwise_conditioned_k_100')
    pd.DataFrame([base] if not drop else [], columns=['recording', 'model']).to_csv(
        out/'v2_instantaneous_null100_triplet_residuals.csv', index=False)
    pd.DataFrame([dict(**base, metric='h1', n_null_blocks=100)]).to_csv(
        out/'v2_instantaneous_null100_comparisons.csv', index=False)
    pd.DataFrame([dict(recording=a, n_requested_nulls=100, n_valid_nulls=100)]).to_csv(
        out/'v2_instantaneous_null100_qc.csv', index=False)


def test_explicit_manifest_controls_current_null_n(tmp_path):
    write_fixture(tmp_path)
    residual, topology = load_instantaneous_null100(tmp_path)
    assert len(residual) == len(topology) == 1


def test_missing_eligible_residual_is_not_silently_dropped(tmp_path):
    write_fixture(tmp_path, drop=True)
    with pytest.raises(ValueError):
        load_instantaneous_null100(tmp_path)
