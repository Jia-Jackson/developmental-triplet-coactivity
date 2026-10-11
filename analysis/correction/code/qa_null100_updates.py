"""Numerical provenance and unchanged-output audit plus visual contact sheets."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from hotopology.current_null import load_instantaneous_null100

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / 'tables/computational_closure_v2'
QA = ROOT / 'qa/null100_unification_20260907'
FIG = Path(r'D:\Published paper\high-order-topology\CSF_prepartion\Figure')
PANELS = {
    'Figure.3': ['Figure_3D_covariate_adjusted_effects_compact'],
    'Figure.6': ['Figure_6A_independent_culture_direction', 'Figure_6B_hierarchical_early_effects', 'Figure_6D_node_specification_sensitivity'],
    'Figure.S6': ['Figure_S6A_hierarchical_effects', 'Figure_S6B_culture_specific_median_change'],
    'Figure.S9': ['Figure_S9A_activity_form_selection', 'Figure_S9B_tier1_vs_tier2', 'Figure_S9C_activity_dependence'],
    'Figure.S10': ['Figure_S10A_late_range_prediction', 'Figure_S10B_late_stage_bootstrap', 'Figure_S10C_triplet_participation'],
}


def main():
    residual, topology = load_instantaneous_null100(ROOT)
    activity = pd.read_csv(V2 / 'v2_activity_adjustment_analysis_table.csv').set_index('recording').sort_index()
    reference = residual.set_index('recording').sort_index()
    for column in ['absolute_residual_z', 'positive_residual_z', 'absolute_residual_mass']:
        np.testing.assert_allclose(activity[column], reference[column], rtol=1e-12)
    for outcome, metric in [('h1_persistence_deviation_z', 'h1_total_persistence_capped'), ('h1_infinite_deviation_z', 'h1_infinite_count')]:
        expected = topology.loc[topology.metric.eq(metric)].set_index('recording').sort_index()
        np.testing.assert_allclose(activity[outcome], expected['z'], rtol=1e-12)
    for file in ['v2_hierarchical_adjacent_contrasts.csv', 'v2_hierarchical_coefficients.csv', 'v2_culture_specific_adjacent_changes.csv']:
        old = pd.read_csv(QA / 'tables_before' / file)
        new = pd.read_csv(V2 / file)
        keep = ['direct_25ms', 'burst_inside_outside_difference']
        pd.testing.assert_frame_equal(old[old.family.isin(keep)].reset_index(drop=True), new[new.family.isin(keep)].reset_index(drop=True))
    incremental = pd.read_csv(V2 / 'v2_incremental_analysis_table.csv').set_index('recording').sort_index()
    for column in ['absolute_residual_z', 'positive_residual_z', 'log1p_absolute_residual_mass', 'h1_persistence_deviation_z', 'h1_infinite_deviation_z']:
        np.testing.assert_allclose(incremental[column], activity[column], rtol=1e-12)
    audit = pd.read_json(V2 / 'v2_incremental_fold_audit.json')
    assert audit.outer_groups_disjoint.all()
    # Fresh prediction must postdate its updated inputs, preventing stale-output approval.
    assert (V2 / 'v2_incremental_metrics.csv').stat().st_mtime > (V2 / 'v2_activity_adjustment_analysis_table.csv').stat().st_mtime
    allowed = {str(FIG / folder / (stem + suffix)) for folder, stems in PANELS.items() for stem in stems for suffix in ['.png', '.pdf', '.svg', '.tiff']}
    changed = []
    for row in pd.read_csv(QA / 'figure_hashes_before.csv').itertuples(index=False):
        path = Path(row.Path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        if digest != row.Hash:
            assert str(path) in allowed, f'Unintended figure mutation: {path}'
            changed.append(dict(path=str(path), old_sha256=row.Hash, new_sha256=digest))
    pd.DataFrame(changed).to_csv(QA / 'changed_figure_files.csv', index=False)
    images = []
    for folder, stems in PANELS.items():
        for stem in stems:
            path = FIG / folder / f'{stem}.png'
            with Image.open(path) as source:
                im = source.convert('RGB')
                im.thumbnail((750, 570))
                tile = Image.new('RGB', (800, 620), 'white')
                tile.paste(im, ((800-im.width)//2, 35+(570-im.height)//2))
                ImageDraw.Draw(tile).text((15, 10), stem, fill='black')
                images.append(tile)
    for first in range(0, len(images), 6):
        sheet = Image.new('RGB', (2400, 1240), '#dddddd')
        for index, tile in enumerate(images[first:first+6]):
            sheet.paste(tile, ((index%3)*800, (index//3)*620))
        sheet.save(QA / f'contact_sheet_{first//6+1}.png')
    result = dict(n_recordings=len(activity), current_null_realizations=100, unchanged_direct_and_burst_statistics=True,
                  prediction_inputs_match=True, prediction_outer_folds_disjoint=True, n_changed_figure_files=len(changed),
                  no_unintended_figure_or_ppt_changes=True)
    (QA / 'verification_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
