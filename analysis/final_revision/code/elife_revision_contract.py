"""Shared scientific and editorial contracts for the corrected eLife package."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

import pandas as pd


VALID_RESIDUAL_TOPOLOGICAL_FEATURES: tuple[str, ...] = (
    "log1p_absolute_residual_mass",
    "absolute_residual_z",
    "h1_persistence_deviation_z",
    "direct_triplet_repeated5_density",
    "direct_h1_total_persistence_capped",
)


def apply_endpoint_z_policy(
    frame: pd.DataFrame,
    *,
    sd_column: str = "null_sd",
    z_column: str = "z",
) -> pd.DataFrame:
    """Mark every zero-variance standardized endpoint score as undefined.

    A z score requires a nonzero reference standard deviation.  The raw
    observation, reference mean, and empirical tail probability remain valid
    and are intentionally left untouched.
    """

    corrected = frame.copy()
    defined = corrected[sd_column].notna() & corrected[sd_column].gt(0)
    corrected["z_defined"] = defined
    corrected.loc[~defined, z_column] = pd.NA
    return corrected


def assert_parent_linked_supplement_map(mapping: Mapping[str, str]) -> None:
    """Validate the twelve eLife parent-linked figure-supplement labels."""

    if set(mapping) != {f"S{i}" for i in range(1, 13)}:
        raise ValueError("Supplement mapping must define S1 through S12 exactly once")
    values = list(mapping.values())
    if len(values) != len(set(values)):
        raise ValueError("Every source supplement must map to a unique output label")
    pattern = re.compile(r"^Figure [1-7]—figure supplement [1-9]\d*$")
    invalid = [value for value in values if not pattern.fullmatch(value)]
    if invalid:
        raise ValueError(f"Invalid eLife supplement labels: {invalid}")


def canonicalize_reference_numbers(
    reference_numbers: Iterable[int],
    duplicate_to_canonical: Mapping[int, int],
) -> dict[int, int]:
    """Map old bibliography numbers onto a de-duplicated sequential list."""

    numbers = list(reference_numbers)
    surviving = [number for number in numbers if number not in duplicate_to_canonical]
    renumber = {old: new for new, old in enumerate(surviving, start=1)}
    result: dict[int, int] = {}
    for old in numbers:
        canonical = duplicate_to_canonical.get(old, old)
        if canonical not in renumber:
            raise ValueError(f"Duplicate {old} points to missing canonical reference {canonical}")
        result[old] = renumber[canonical]
    return result

