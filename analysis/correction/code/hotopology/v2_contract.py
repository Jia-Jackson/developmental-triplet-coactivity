from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
from typing import Iterable

import pandas as pd


_RECORDING_RE = re.compile(
    r"^WT_DIV(?P<div>\d+)_P(?P<plate>\d+)_(?P<well>[AB]\d)$"
)


def _first_present(table: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in table.columns:
            return name
    return None


def normalize_recording_keys(table: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical recording, DIV, plate, well, and subject keys."""
    result = table.copy()
    recording_col = _first_present(result, ["recording", "SourceRecording", "Recording"])
    if recording_col is None:
        raise ValueError("recording identifier column not found")
    result["recording"] = result[recording_col].astype(str).str.strip()

    parsed = result["recording"].str.extract(_RECORDING_RE)
    if parsed.isna().any(axis=None):
        bad = result.loc[parsed.isna().any(axis=1), "recording"].tolist()[:5]
        raise ValueError(f"unrecognized recording identifiers: {bad}")

    div_col = _first_present(result, ["DIV", "div"])
    plate_col = _first_present(result, ["plate", "PlateIndex", "plate_index"])
    well_col = _first_present(result, ["well", "Well"])
    result["DIV"] = (
        pd.to_numeric(result[div_col], errors="raise").astype(int)
        if div_col
        else parsed["div"].astype(int)
    )
    result["plate"] = (
        pd.to_numeric(result[plate_col], errors="raise").astype(int)
        if plate_col
        else parsed["plate"].astype(int)
    )
    result["well"] = (
        result[well_col].astype(str).str.strip() if well_col else parsed["well"]
    )

    canonical = (
        "WT_DIV"
        + result["DIV"].astype(str)
        + "_P"
        + result["plate"].astype(str)
        + "_"
        + result["well"]
    )
    mismatch = result["recording"] != canonical
    if mismatch.any():
        raise ValueError(
            "recording metadata mismatch: "
            + ", ".join(result.loc[mismatch, "recording"].head(5))
        )
    result["recording_key"] = canonical
    result["culture"] = "P" + result["plate"].astype(str)
    result["subject_id"] = result["culture"] + "_" + result["well"]
    return result


def expected_recording_grid(
    *,
    divs: Iterable[int],
    cultures: Iterable[int],
    wells: Iterable[str],
    excluded_recording: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for div in divs:
        for plate in cultures:
            for well in wells:
                recording = f"WT_DIV{int(div)}_P{int(plate)}_{well}"
                if recording == excluded_recording:
                    continue
                rows.append(
                    {
                        "recording": recording,
                        "DIV": int(div),
                        "plate": int(plate),
                        "well": str(well),
                        "culture": f"P{int(plate)}",
                        "subject_id": f"P{int(plate)}_{well}",
                    }
                )
    return pd.DataFrame(rows)


def audit_recording_table(observed: pd.DataFrame, expected: pd.DataFrame) -> dict[str, object]:
    normalized = normalize_recording_keys(observed)
    expected_normalized = normalize_recording_keys(expected)
    if normalized["recording"].duplicated().any():
        duplicates = normalized.loc[
            normalized["recording"].duplicated(keep=False), "recording"
        ].unique()
        raise ValueError(f"duplicate recording keys: {duplicates[:5].tolist()}")
    observed_keys = set(normalized["recording"])
    expected_keys = set(expected_normalized["recording"])
    missing = sorted(expected_keys - observed_keys)
    unexpected = sorted(observed_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(f"missing={missing[:5]}; unexpected={unexpected[:5]}")
    return {
        "n_rows": int(len(normalized)),
        "n_unique_recordings": int(normalized["recording"].nunique()),
        "n_subjects": int(normalized["subject_id"].nunique()),
        "n_cultures": int(normalized["culture"].nunique()),
        "div_counts": {
            str(int(k)): int(v)
            for k, v in normalized.groupby("DIV", sort=True).size().items()
        },
    }


def audited_one_to_one_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    suffixes: tuple[str, str] = ("", "_right"),
) -> pd.DataFrame:
    left_n = normalize_recording_keys(left)
    right_n = normalize_recording_keys(right)
    for label, table in [("left", left_n), ("right", right_n)]:
        if table["recording"].duplicated().any():
            raise ValueError(f"duplicate recording keys in {label} table")
    merged = left_n.merge(
        right_n,
        on="recording",
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=suffixes,
    )
    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        keys = merged.loc[unmatched, ["recording", "_merge"]].head(10).to_dict("records")
        raise ValueError(f"one-to-one merge has unmatched keys: {keys}")
    return merged.drop(columns="_merge")


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
