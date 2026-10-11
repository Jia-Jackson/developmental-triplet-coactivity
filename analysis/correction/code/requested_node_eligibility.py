"""Explicit requested-N eligibility; no reduced-N substitutes or record deletion."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from hotopology.node_selection import rank_fixed_longitudinal_nodes


def eligible_nodes(x):
    x = np.asarray(x)
    if x.ndim != 2 or not np.isin(x, [0, 1]).all():
        raise ValueError('Expected two-dimensional binary activity')
    counts = x.sum(axis=1, dtype=np.int64)
    nodes = np.flatnonzero((counts >= 20) & (x.shape[1] - counts >= 20))
    return nodes[np.lexsort((nodes, -counts[nodes]))]


def fixed_sets(arrays, node_counts):
    ranking = rank_fixed_longitudinal_nodes(arrays, min_active_bins=20, min_inactive_bins=20)
    nodes = ranking['electrode_index_zero_based'].to_numpy(dtype=int)
    return {n: nodes[:n] for n in node_counts if len(nodes) >= n}, ranking


def dynamic_model_nodes(x, n):
    nodes = eligible_nodes(x)
    return np.sort(nodes[:n]) if len(nodes) >= n else None


def expected_keys(frame, analysis):
    if frame.duplicated(['recording', 'analysis']).any():
        raise ValueError('Duplicate eligibility entries')
    subset = frame.loc[frame.analysis.eq(analysis)]
    if subset.empty:
        raise ValueError('Missing requested analysis eligibility')
    valid = subset.eligible.map(lambda v: str(v).lower() == 'true')
    return set(subset.loc[valid, 'recording'])


def build(root):
    root = Path(root)
    manifest = pd.read_csv(root / 'tables/input_provenance.csv')
    manifest = manifest.loc[manifest.excluded.map(lambda v: str(v).lower() == 'false')]
    paths = sorted((root / 'data_intermediate/binary_25ms').glob('*_25ms.npz'))
    by_subject, records = {}, {}
    for path in paths:
        recording = path.stem.removesuffix('_25ms')
        _, div_text, culture, well = recording.split('_')
        div = int(div_text.removeprefix('DIV'))
        with np.load(path) as archive:
            x = np.asarray(archive['x'], dtype=np.uint8)
        records[recording] = (div, culture, well, x)
        by_subject.setdefault((culture, well), {})[div] = x
    if len(records) != 143 or set(records) != set(manifest.recording):
        raise ValueError('Binary record keys differ from verified retained inputs')
    rankings = {subject: fixed_sets(arrays, [12, 16, 20])[1]
                for subject, arrays in by_subject.items()}
    rows = []
    for recording, (div, culture, well, x) in records.items():
        dynamic = len(eligible_nodes(x))
        common = len(rankings[culture, well])
        for analysis, requested, available in [('full_array', 0, x.shape[0]),
                ('dynamic16', 16, dynamic), ('fixed12', 12, common),
                ('fixed16', 16, common), ('fixed20', 20, common)]:
            valid = available >= requested
            rows.append(dict(recording=recording, DIV=div, culture=culture, well=well,
                subject_id=f'{culture}_{well}', analysis=analysis, requested_n=requested,
                eligible_nodes=available, eligible=valid, n_bins=x.shape[1],
                reason='' if valid else 'insufficient_eligible_nodes_for_requested_N'))
    frame = pd.DataFrame(rows)
    out = root / 'tables/computational_closure_v2'
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'requested_node_eligibility.csv', index=False)
    counts = frame.groupby(['analysis', 'DIV', 'culture']).agg(
        n_available=('eligible', 'size'), n_eligible=('eligible', 'sum')).reset_index()
    counts.to_csv(out / 'requested_node_sample_sizes.csv', index=False)
    totals = {name: len(expected_keys(frame, name)) for name in frame.analysis.unique()}
    expected = dict(full_array=143, dynamic16=142, fixed12=143, fixed16=135, fixed20=127)
    if totals != expected:
        raise ValueError(f'Eligibility differs from approved preflight: {totals}')
    (root / 'qa').mkdir(exist_ok=True)
    (root / 'qa/requested_node_eligibility.json').write_text(
        json.dumps(dict(status='passed', counts=totals, excluded_records=[]), indent=2), encoding='utf-8')
    print(json.dumps(totals), flush=True)
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=Path, required=True)
    build(parser.parse_args().result_dir)
