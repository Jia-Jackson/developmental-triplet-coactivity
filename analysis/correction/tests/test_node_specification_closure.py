import pandas as pd
import pytest
from run_v2_node_robustness import require_recording_grid

def test_missing_extra_and_duplicate_rows_are_rejected():
    require_recording_grid(pd.DataFrame({'recording': ['a', 'b']}), {'a','b'}, 'test')
    for keys in [['a'], ['a','b','c'], ['a','a','b']]:
        with pytest.raises(ValueError):
            require_recording_grid(pd.DataFrame({'recording': keys}), {'a','b'}, 'test')

def test_long_metric_grid_requires_each_recording_per_metric():
    table=pd.DataFrame({'recording':['a','b','a','b'], 'metric':['x','x','y','y']})
    require_recording_grid(table, {'a','b'}, 'long', metric_col='metric')
    with pytest.raises(ValueError):
        require_recording_grid(table.iloc[:3], {'a','b'}, 'long', metric_col='metric')
