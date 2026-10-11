import pandas as pd
import pytest
from run_v2_compile_results import require_qc_cohort, require_age_model_grid


def test_qc_coverage_rejects_missing_eligible_record():
    frame=pd.DataFrame([dict(recording='a',n_requested_nulls=100,n_valid_nulls=100)])
    with pytest.raises(ValueError):
        require_qc_cohort(frame, {'a','b'}, 'test')
    assert require_qc_cohort(frame, {'a'}, 'test')['n_valid100']==1


def test_age_grid_does_not_accept_vacuous_partial_success():
    frame=pd.DataFrame([dict(model='m',DIV=14)])
    with pytest.raises(ValueError):
        require_age_model_grid(frame)
