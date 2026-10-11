from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))


def test_hierarchical_residual_uses_current_100_realizations():
    from run_v2_hierarchical_statistics import _assemble_datasets
    actual = _assemble_datasets(ROOT)['conditioned_triplet_residual'][0]
    expected = pd.read_csv(ROOT / 'tables/computational_closure_v2/v2_instantaneous_null100_triplet_residuals.csv')
    actual = actual.set_index('recording').sort_index()
    expected = expected.set_index('recording').sort_index()
    assert set(actual['model']) == {'pairwise_conditioned_k_100'}
    np.testing.assert_allclose(actual['absolute_residual_z'], expected['absolute_residual_z'], rtol=1e-12)


def test_figure6d_distinguishes_raw_and_null_normalized_h1():
    path = Path(r'D:\Published paper\high-order-topology\CSF_prepartion\Figure\Figure.6\export_figure6d_adjusted.py')
    spec = importlib.util.spec_from_file_location('figure6d_regression', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.load_data()
    assert 'display_key' in data.columns
    assert data.groupby('node_specification')['display_key'].nunique().eq(4).all()
    h1 = data[data['outcome'].eq('h1_total_persistence_capped')]
    assert h1['display_key'].nunique() == 2


def test_figure6b_extreme_marker_has_right_margin():
    path = ROOT / 'Chaos, Solitons & Fractals/code/build_csf_figures.py'
    spec = importlib.util.spec_from_file_location('csf_margin_regression', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.load_original_module()
    captured = []
    module.export_full = lambda fig, *args: captured.append(fig)
    module.export_panels = lambda *args: None
    # Source CSV belongs to a temporary test directory, not production outputs.
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        module.SOURCE_DATA = Path(temp)
        module.build_figure_6(original)
    fig = captured[0]
    axis = fig.axes[1]
    fig.canvas.draw()
    maximum = max(float(collection.get_offsets()[:, 0].max()) for collection in axis.collections)
    x_pixel = axis.transData.transform((maximum, 0))[0]
    assert axis.bbox.x1 - x_pixel > 6, 'Rightmost marker is clipped at the axis boundary'
