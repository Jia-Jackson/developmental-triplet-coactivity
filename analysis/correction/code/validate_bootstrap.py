"""Read-only validation of completed bootstrap outputs, plus safe missing-code copies."""
from pathlib import Path
import hashlib
import json
import pandas as pd

from bootstrap_inputs import (OLD, RAW_ROOT, ROOT, copy_verified, iter_original_code_files,
                              sha256, validate_duration_bins, validate_raw_source,
                              validate_reuse_artifacts)

SAFETY_LOG = ROOT / 'logs/bootstrap_safety_validation.json'
SUPPLEMENTAL = ROOT / 'inputs/source_copy_manifest_supplemental.csv'

def collection_digest(records):
    canonical = '\n'.join(f'{r[0]}\t{r[1]}' for r in sorted(records))
    return hashlib.sha256(canonical.encode('utf8')).hexdigest()

def verify_manifest(path, source_root=None, destination_root=None, require_relative_identity=False):
    manifest = pd.read_csv(path)
    required = {'source', 'destination', 'sha256', 'bytes'}
    if set(manifest.columns) != required:
        raise ValueError(f'Unexpected manifest columns: {path}')
    hashes = []
    source_root = Path(source_root).resolve() if source_root is not None else None
    destination_root = Path(destination_root).resolve() if destination_root is not None else None
    for row in manifest.itertuples(index=False):
        source, destination = Path(row.source).resolve(), Path(row.destination).resolve()
        if source_root is not None and not source.is_relative_to(source_root):
            raise ValueError(f'Manifest source outside expected root: {source}')
        if destination_root is not None and not destination.is_relative_to(destination_root):
            raise ValueError(f'Manifest destination outside expected root: {destination}')
        if require_relative_identity and source.relative_to(source_root) != destination.relative_to(destination_root):
            raise ValueError(f'Manifest source/destination identity mismatch: {source} -> {destination}')
        if not source.is_file() or not destination.is_file():
            raise FileNotFoundError(f'Manifest file missing: {source} -> {destination}')
        source_hash, destination_hash = sha256(source), sha256(destination)
        if source_hash != row.sha256 or destination_hash != row.sha256:
            raise RuntimeError(f'Manifest hash mismatch: {destination}')
        if source.stat().st_size != row.bytes or destination.stat().st_size != row.bytes:
            raise RuntimeError(f'Manifest size mismatch: {destination}')
        hashes.append((str(destination.resolve()), destination_hash))
    return manifest, hashes

def validate_retained_grid(retained):
    wells = ('A1', 'A2', 'A3', 'B1', 'B2', 'B3')
    expected = {(f'WT_DIV{div}_P{plate}_{well}', div, plate, well)
                for div in range(14, 64, 7) for plate in (1, 2, 3) for well in wells
                if f'WT_DIV{div}_P{plate}_{well}' != 'WT_DIV35_P2_A1'}
    actual = {(str(r.recording), int(r.DIV), int(r.plate), str(r.well))
              for r in retained[['recording', 'DIV', 'plate', 'well']].itertuples(index=False)}
    if len(retained) != len(actual) or actual != expected:
        raise ValueError(f'Retained recording grid mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}')
    return {recording for recording, _, plate, _ in expected if plate in (1, 2)}

def main():
    for output in (SAFETY_LOG, SUPPLEMENTAL):
        if output.exists():
            raise FileExistsError(f'Refusing to replace existing validation output: {output}')

    original_manifest_path = ROOT / 'inputs/source_copy_manifest.csv'
    original_manifest, copied_hashes = verify_manifest(
        original_manifest_path, destination_root=ROOT)

    provenance = pd.read_csv(ROOT / 'tables/input_provenance.csv')
    raw_hashes = []
    for raw_path, expected_hash in provenance[['raw_path', 'raw_sha256']].drop_duplicates().itertuples(index=False):
        resolved = validate_raw_source(raw_path, RAW_ROOT)
        actual_hash = sha256(resolved)
        if actual_hash != expected_hash:
            raise RuntimeError(f'Raw source hash mismatch: {resolved}')
        raw_hashes.append((str(resolved), actual_hash))
    if len(raw_hashes) != 24:
        raise ValueError(f'Expected 24 unique raw files, found {len(raw_hashes)}')

    retained = provenance[~provenance.excluded.astype(bool)]
    p1p2_ids = validate_retained_grid(retained)

    reuse_path = ROOT / 'inputs/p1p2_local_artifact_reuse.csv'
    reuse_manifest, reuse_hashes = verify_manifest(
        reuse_path, OLD/'data_intermediate', ROOT/'data_intermediate', True)
    identities = [(Path(row.destination).parent.name, Path(row.destination).name)
                  for row in reuse_manifest.itertuples(index=False)]
    validate_reuse_artifacts(identities, p1p2_ids, require_complete=True)

    duration_evidence = validate_duration_bins(retained)

    supplemental = []
    original_code = iter_original_code_files(OLD / 'code')
    for source in original_code:
        destination = ROOT / 'code' / source.relative_to(OLD / 'code')
        if destination.exists():
            if not destination.is_file() or sha256(destination) != sha256(source):
                raise FileExistsError(f'Conflicting original-code destination: {destination}')
        else:
            supplemental.append(copy_verified(source, destination, ROOT))
    pd.DataFrame(supplemental, columns=['source', 'destination', 'sha256', 'bytes']).to_csv(
        SUPPLEMENTAL, index=False, mode='x')

    code_hashes = [(str((ROOT / 'code' / p.relative_to(OLD / 'code')).resolve()), sha256(p))
                   for p in original_code]
    evidence = {
        'status': 'passed',
        'source_writes': False,
        'full_bootstrap_recomputed': False,
        'original_manifest': {'path': str(original_manifest_path), 'rows': int(len(original_manifest)),
                              'sha256': sha256(original_manifest_path),
                              'verified_destination_collection_sha256': collection_digest(copied_hashes)},
        'supplemental_code_manifest': {'path': str(SUPPLEMENTAL), 'rows': len(supplemental),
                                       'files': [Path(x['destination']).name for x in supplemental],
                                       'sha256': sha256(SUPPLEMENTAL)},
        'original_code_tree': {'source_files': len(original_code),
                               'verified_file_collection_sha256': collection_digest(code_hashes)},
        'raw_scope': {'approved_root': str(RAW_ROOT.resolve()), 'unique_files': len(raw_hashes),
                      'verified_file_collection_sha256': collection_digest(raw_hashes)},
        'retained_records': int(len(retained)),
        'retained_p1p2_recordings': len(p1p2_ids),
        'reuse_allowlist': {'rows': int(len(reuse_manifest)),
                            'verified_file_collection_sha256': collection_digest(reuse_hashes)},
        'duration_25ms_bins': duration_evidence,
    }
    SAFETY_LOG.write_text(json.dumps(evidence, indent=2), encoding='utf8', errors='strict')
    print(json.dumps(evidence, indent=2))

if __name__ == '__main__':
    main()
