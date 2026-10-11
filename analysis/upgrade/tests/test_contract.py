
from release_paths import repository_path
from pathlib import Path

import pandas as pd

from higher_order_v2.contract import audit_fixed12_contract


SOURCE = repository_path('analysis/correction')


def test_corrected_fixed12_contract_has_17_complete_wells():
    manifest = pd.read_csv(SOURCE / "tables/computational_closure_v2/v2_fixed_node_manifest_long.csv")
    recording_keys = pd.read_csv(SOURCE / "tables/computational_closure_v2/v2_recording_key_manifest.csv")
    audit = audit_fixed12_contract(manifest, recording_keys)
    assert audit.n_recordings == 143
    assert audit.n_complete_wells == 17
    assert audit.complete_wells_by_culture == {"P1": 6, "P2": 5, "P3": 6}
    assert "P2_A1" not in audit.complete_subjects
    assert audit.nodes_constant_within_subject

