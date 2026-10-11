"""Validated, shared source for current instantaneous-null downstream analyses."""
from pathlib import Path
import pandas as pd
from .v2_contract import normalize_recording_keys
from requested_node_eligibility import expected_keys

MODEL = 'pairwise_conditioned_k_100'


def load_instantaneous_null100(root: Path):
    folder = Path(root) / 'tables' / 'computational_closure_v2'
    residual = pd.read_csv(folder / 'v2_instantaneous_null100_triplet_residuals.csv')
    topology = pd.read_csv(folder / 'v2_instantaneous_null100_comparisons.csv')
    qc = pd.read_csv(folder / 'v2_instantaneous_null100_qc.csv')
    for frame in (residual, topology):
        if set(frame['model']) != {MODEL}:
            raise ValueError('Current downstream analysis requires the instantaneous 100-realization model')
    if not topology['n_null_blocks'].eq(100).all():
        raise ValueError('Every current topology comparison must use 100 valid realizations')
    if not qc[['n_requested_nulls', 'n_valid_nulls']].eq(100).all().all():
        raise ValueError('Current null QC must verify 100 requested and valid realizations')
    keys = set(qc['recording'])
    eligibility = pd.read_csv(folder / 'requested_node_eligibility.csv')
    required_keys = expected_keys(eligibility, 'dynamic16')
    if keys != required_keys or qc['recording'].duplicated().any():
        raise ValueError('QC recording set differs from explicit dynamic16 eligibility manifest')
    if set(residual['recording']) != keys or set(topology['recording']) != keys:
        raise ValueError('Null source recording sets do not match')
    if residual['recording'].duplicated().any() or topology.duplicated(['recording', 'metric']).any():
        raise ValueError('Duplicate null source entries')
    residual['n_null_realizations'] = 100
    return normalize_recording_keys(residual), normalize_recording_keys(topology)
