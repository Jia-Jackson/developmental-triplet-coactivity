import numpy as np
import pytest
from requested_node_eligibility import eligible_nodes, fixed_sets, expected_keys, dynamic_model_nodes


def binary(n):
    x = np.zeros((24, 60), dtype=np.uint8)
    x[:n, :20] = 1
    return x


def test_dynamic_exact_n_never_shrinks_or_pads():
    assert len(eligible_nodes(binary(14))) == 14
    assert list(eligible_nodes(binary(18))) == list(range(18))


def test_dynamic_model_order_matches_original_matlab_electrode_sort():
    x = binary(24)
    x[20:24, :30] = 1
    assert np.array_equal(dynamic_model_nodes(x, 16), np.r_[np.arange(12), np.arange(20,24)])
    assert dynamic_model_nodes(binary(14), 16) is None


def test_fixed_availability_is_separate_for_each_requested_size():
    selected, ranking = fixed_sets({14: binary(18), 21: binary(13)}, [12, 16, 20])
    assert set(selected) == {12}
    assert len(ranking) == 13
    assert np.array_equal(selected[12], np.arange(12))


def test_fixed_sets_preserve_original_nested_rank():
    selected, ranking = fixed_sets({14: binary(24), 21: binary(24)}, [12, 16, 20])
    assert np.array_equal(selected[12], selected[20][:12])
    assert len(ranking) == 24


def test_requested_key_contract_rejects_duplicate_rows():
    import pandas as pd
    rows = pd.DataFrame([dict(recording='r', analysis='dynamic16', eligible=True)] * 2)
    with pytest.raises(ValueError, match='Duplicate'):
        expected_keys(rows, 'dynamic16')


def test_requested_key_contract_excludes_only_specific_analysis():
    import pandas as pd
    rows = pd.DataFrame([
        dict(recording='r', analysis='dynamic16', eligible=False),
        dict(recording='r', analysis='full_array', eligible=True),
        dict(recording='s', analysis='dynamic16', eligible=True),
    ])
    assert expected_keys(rows, 'dynamic16') == {'s'}
    assert expected_keys(rows, 'full_array') == {'r'}
