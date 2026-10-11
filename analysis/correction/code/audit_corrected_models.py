"""Check completed MATLAB outputs against exact node and recording manifests."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from requested_node_eligibility import dynamic_model_nodes, expected_keys


def audit(root, analysis):
    root = Path(root)
    table_dir = root / 'tables/computational_closure_v2'
    eligibility = pd.read_csv(table_dir / 'requested_node_eligibility.csv')
    keys = expected_keys(eligibility, analysis)
    n = int(analysis.replace('dynamic', '').replace('fixed', ''))
    folder = 'maxent_models_top16' if analysis == 'dynamic16' else f'maxent_models_fixed_n{n}'
    paths = list((root / 'data_intermediate' / folder).glob('*_maxent.mat'))
    if {p.stem.removesuffix('_maxent') for p in paths} != keys:
        raise ValueError(f'{analysis}: model set incomplete or contains ineligible records ({len(paths)}/{len(keys)})')
    fixed = pd.read_csv(table_dir / 'v2_fixed_node_manifest_long.csv')
    rows = []
    for path in sorted(paths):
        recording = path.stem.removesuffix('_maxent')
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        if bool(mat['skipped']):
            raise ValueError(f'Skipped stub: {recording}')
        fit = mat['fit']
        nodes = np.asarray(mat['active_nodes']).astype(int).reshape(-1) - 1
        with np.load(root/'data_intermediate/binary_25ms'/f'{recording}_25ms.npz') as z:
            full = z['x']
        selected = (dynamic_model_nodes(full, n) if analysis.startswith('dynamic') else
            fixed.loc[fixed.recording.eq(recording) & fixed.n_nodes.eq(n)].sort_values('fixed_rank').electrode_index_zero_based.to_numpy(int))
        if not np.array_equal(nodes, selected):
            raise ValueError(f'Node selection mismatch: {recording}/{analysis}')
        if int(fit.n_nodes) != n or int(fit.n_data_bins) != full.shape[1]:
            raise ValueError(f'Input shape mismatch: {recording}')
        pool = np.asarray(fit.pairwise_samples)
        if pool.shape != (n, 20 * full.shape[1]) or not np.isin(pool, [0, 1]).all():
            raise ValueError(f'Invalid 20-block source pool: {recording}')
        rows.append(dict(recording=recording, analysis=analysis, n_nodes=n,
            n_bins=full.shape[1], converged=bool(fit.pairwise_converged),
            model_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    out = pd.DataFrame(rows)
    out.to_csv(table_dir / f'{analysis}_model_audit.csv', index=False)
    summary = dict(status='passed', analysis=analysis, n_models=len(out),
        n_converged=int(out.converged.sum()), exact_node_and_duration_matches=True)
    (root/'qa'/f'{analysis}_model_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--analysis', required=True)
    args = parser.parse_args()
    audit(args.result_dir, args.analysis)
