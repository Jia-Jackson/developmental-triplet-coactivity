from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_savez(path: Path, **arrays: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def parse_recording(recording: str) -> tuple[int, str, str, str]:
    parts = recording.split("_")
    if len(parts) != 4 or not parts[1].startswith("DIV") or not parts[2].startswith("P"):
        raise ValueError(f"unrecognized recording id: {recording}")
    div = int(parts[1].removeprefix("DIV"))
    culture = parts[2]
    well = parts[3]
    return div, culture, well, f"{culture}_{well}"

