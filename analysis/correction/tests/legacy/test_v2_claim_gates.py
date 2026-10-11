from __future__ import annotations


def test_incremental_gate_distinguishes_supported_partial_and_not_supported():
    from hotopology.claim_gates import incremental_value_gate

    assert incremental_value_gate(1.2, 0.3, 2.0, 3, 3)["status"] == "supported"
    assert incremental_value_gate(0.4, -0.2, 1.0, 2, 3)["status"] == "partially_supported"
    assert incremental_value_gate(-0.1, -0.7, 0.3, 1, 3)["status"] == "not_supported"


def test_robustness_gate_marks_direction_reversal_as_not_supported():
    from hotopology.claim_gates import robustness_gate

    assert robustness_gate([1.0, 0.8, 1.2], minimum_agreement=0.75)["status"] == "supported"
    assert robustness_gate([1.0, -0.8, -1.2], minimum_agreement=0.75)["status"] == "not_supported"


def test_technical_gate_is_inconclusive_when_null_count_is_low():
    from hotopology.claim_gates import null_technical_gate

    assert null_technical_gate(valid_nulls=89, minimum_valid=90, qc_pass=True)["status"] == "technically_inconclusive"
    assert null_technical_gate(valid_nulls=100, minimum_valid=90, qc_pass=True)["status"] == "supported"
