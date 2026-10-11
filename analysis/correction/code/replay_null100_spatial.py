"""Replay the existing 100-null seed for exploratory triplet participation only.

No fitting or topological analysis is repeated. Each reconstructed absolute
residual mean must match the previously published 100-realization summary.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
import json
import zlib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from hotopology.null_models import decode_ising_factors, resample_pool_conditioned_on_k
from hotopology.simplicial import simplex_supports
from hotopology.current_null import load_instantaneous_null100

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / 'tables/computational_closure_v2'


def replay_one(path_string):
    path = Path(path_string)
    recording = path.stem.removesuffix('_maxent')
    model = loadmat(path, squeeze_me=True, struct_as_record=False)
    nodes = np.asarray(model['active_nodes'], dtype=int).reshape(-1) - 1
    fit = model['fit']
    with np.load(ROOT / 'data_intermediate/binary_25ms' / f'{recording}_25ms.npz') as archive:
        observed = np.asarray(archive['x'], dtype=np.uint8)[nodes]
        div, plate, well = int(archive['div']), int(archive['plate']), str(archive['well'])
    n_nodes, n_bins = observed.shape
    pool = np.asarray(fit.pairwise_samples, dtype=np.uint8).reshape(n_nodes, -1)
    h, j = decode_ising_factors(np.asarray(fit.pairwise_model.factors), n_nodes)
    simulated = resample_pool_conditioned_on_k(
        pool, observed.sum(axis=0, dtype=np.int32), n_surrogates=100,
        seed=int(zlib.crc32((recording + '_instantaneous100_v2').encode('utf-8'))),
        fallback_h=h, fallback_j=j, fallback_sweeps=20,
    )
    if not np.array_equal(simulated.sum(axis=1), np.broadcast_to(observed.sum(axis=0), (100, n_bins))):
        raise ValueError(f'K(t) replay mismatch: {recording}')
    # Integer bit-counts over concatenated null bins equal the sum over realizations.
    merged = simulated.transpose(1, 0, 2).reshape(n_nodes, -1)
    null_support = simplex_supports(merged, max_order=3)[3]
    obs_support = simplex_supports(observed, max_order=3)[3]
    triplets = list(combinations(range(n_nodes), 3))
    residual = np.array([(obs_support.get(t, 0) - null_support.get(t, 0) / 100) / n_bins for t in triplets])
    top = np.argsort(np.abs(residual))[::-1][:400]
    weights = np.zeros(64)
    details = []
    for rank, index in enumerate(top, start=1):
        electrodes = nodes[list(triplets[index])]
        weights[electrodes] += abs(residual[index])
        details.append(dict(recording=recording, DIV=div, rank=rank, electrode_i=int(electrodes[0])+1,
                            electrode_j=int(electrodes[1])+1, electrode_k=int(electrodes[2])+1,
                            residual_probability=float(residual[index]), n_null_realizations=100))
    if weights.sum() > 0:
        weights /= weights.sum()
    rows = [dict(recording=recording, DIV=div, culture=f'P{plate}', well=well,
                 electrode_index=i+1, participation=float(value), n_null_realizations=100)
            for i, value in enumerate(weights)]
    return recording, float(np.mean(np.abs(residual))), rows, details


def main():
    reference, _ = load_instantaneous_null100(ROOT)
    expected = reference.set_index('recording')['absolute_residual_mass']
    files = sorted((ROOT / 'data_intermediate/maxent_models_top16').glob('*_maxent.mat'))
    if {p.stem.removesuffix('_maxent') for p in files} != set(expected.index):
        raise ValueError('Spatial model set differs from validated dynamic16 null cohort')
    all_rows, all_details, checks = [], [], []
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(replay_one, str(path)) for path in files]
        for number, future in enumerate(as_completed(futures), start=1):
            recording, actual, rows, details = future.result()
            target = float(expected.loc[recording])
            np.testing.assert_allclose(actual, target, rtol=1e-11, atol=1e-15,
                                       err_msg=f'100-null replay differs: {recording}')
            checks.append(dict(recording=recording, expected_mass=target, replay_mass=actual,
                               absolute_error=abs(actual-target)))
            all_rows.extend(rows)
            all_details.extend(details)
            if number % 10 == 0 or number == len(files):
                print(f'Validated spatial replay {number}/{len(files)}', flush=True)
    if set(expected.index) != {row['recording'] for row in checks}:
        raise ValueError('Spatial replay incomplete')
    long = pd.DataFrame(all_rows).sort_values(['recording', 'electrode_index'])
    summary = long.groupby(['DIV', 'electrode_index'])['participation'].agg(['count', 'mean', 'sem']).reset_index()
    summary['n_null_realizations'] = 100
    summary.to_csv(V2 / 'v2_instantaneous_null100_spatial_participation.csv', index=False)
    long.to_csv(V2 / 'v2_instantaneous_null100_spatial_recording_values.csv', index=False)
    pd.DataFrame(all_details).sort_values(['recording', 'rank']).to_csv(V2 / 'v2_instantaneous_null100_top400_triplets.csv', index=False)
    pd.DataFrame(checks).sort_values('recording').to_csv(V2 / 'v2_instantaneous_null100_spatial_replay_audit.csv', index=False)
    print(json.dumps({'n_validated_recordings': len(checks), 'max_absolute_mass_error': max(r['absolute_error'] for r in checks)}, indent=2))


if __name__ == '__main__':
    main()
