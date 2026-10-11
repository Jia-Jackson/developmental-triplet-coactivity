from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fixed12Audit:
    n_recordings: int
    n_complete_wells: int
    complete_wells_by_culture: dict[str, int]
    complete_subjects: tuple[str, ...]
    nodes_constant_within_subject: bool


def audit_fixed12_contract(manifest: pd.DataFrame, recording_keys: pd.DataFrame) -> Fixed12Audit:
    fixed = manifest[manifest["n_nodes"] == 12].copy()
    per_recording = fixed.groupby("recording").size()
    if not (per_recording == 12).all():
        raise ValueError("not every fixed-12 recording contains exactly 12 ranked nodes")
    if set(per_recording.index) != set(recording_keys["recording"]):
        raise ValueError("fixed-12 recordings differ from the corrected recording manifest")
    node_tuples = (
        fixed.sort_values(["recording", "fixed_rank"])
        .groupby(["subject_id", "recording"])["electrode_index_zero_based"]
        .apply(tuple)
        .reset_index(name="nodes")
    )
    constant = bool((node_tuples.groupby("subject_id")["nodes"].nunique() == 1).all())
    age_counts = recording_keys.groupby(["culture", "subject_id"])["DIV"].nunique().reset_index()
    complete = age_counts[age_counts["DIV"] == 8].copy()
    by_culture = complete.groupby("culture")["subject_id"].nunique().astype(int).to_dict()
    return Fixed12Audit(
        n_recordings=int(len(recording_keys)),
        n_complete_wells=int(len(complete)),
        complete_wells_by_culture=by_culture,
        complete_subjects=tuple(sorted(complete["subject_id"])),
        nodes_constant_within_subject=constant,
    )

