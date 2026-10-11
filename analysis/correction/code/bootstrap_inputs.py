"""Read-only raw verification and isolated correction-project bootstrap."""

from release_paths import repository_path
from pathlib import Path
import hashlib
import json
import shutil
import sys
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('D:/Published paper/high-order-topology/Data_processing/Result')
CCS = Path('D:/Published paper/Developmental trajectory/New_correction_result_CCS')
EXCLUDED = 'WT_DIV35_P2_A1'
RAW_ROOT = repository_path('raw_spk')
REUSE_VARIANTS = {
    'binary_25ms': ('_25ms.mat', '_25ms.npz'),
    'direct_checkpoints': ('.json',),
    'instantaneous_null100_checkpoints_top16': ('.json',),
    'instantaneous_null100_fixed_n12_checkpoints': ('.json',),
    'instantaneous_null100_fixed_n16_checkpoints': ('.json',),
    'instantaneous_null100_fixed_n20_checkpoints': ('.json',),
    'kinetic_models_top16': ('_kinetic_fit.npz',),
    'kinetic_null_checkpoints_top16': ('.json',),
    'maxent_models_fixed_n12': ('_maxent.mat',),
    'maxent_models_fixed_n16': ('_maxent.mat',),
    'maxent_models_fixed_n20': ('_maxent.mat',),
    'maxent_models_top16': ('_maxent.mat',),
    'null_checkpoints': ('.json',),
    'triplets_25ms': ('_triplets_25ms.npz',),
}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def copy_verified(source, destination, root):
    source, destination, root = Path(source).resolve(), Path(destination).resolve(), Path(root).resolve()
    if not destination.is_relative_to(root) or destination == root:
        raise ValueError('Destination escapes isolated correction root')
    source_hash = sha256(source)
    if destination.exists():
        if not destination.is_file() or sha256(destination) != source_hash:
            raise FileExistsError(f'Refusing to replace different content: {destination}')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as original, destination.open('xb') as output:
            shutil.copyfileobj(original, output, 8*1024*1024)
    if sha256(destination) != source_hash:
        raise RuntimeError(f'Copy readback mismatch: {destination}')
    return {'source': str(source), 'destination': str(destination), 'sha256': source_hash,
            'bytes': destination.stat().st_size}

def _events(times, channels):
    t, c = np.asarray(times, dtype=float).ravel(), np.asarray(channels, dtype=float).ravel()
    if t.size != c.size or not np.isfinite(t).all() or not np.isfinite(c).all():
        raise ValueError('Invalid event arrays')
    if np.any(c != np.floor(c)) or np.any((c < 1) | (c > 64)):
        raise ValueError('Invalid electrode indices')
    order = np.lexsort((t, c))
    return t[order], c[order].astype(np.int16)

def events_equal(first_t, first_c, second_t, second_c):
    a, b = _events(first_t, first_c), _events(second_t, second_c)
    return np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

def iter_original_code_files(code_root):
    """Return every code-tree file except interpreter-generated caches."""
    code_root = Path(code_root)
    return [p for p in sorted(code_root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.relative_to(code_root).parts]

def validate_raw_source(path, approved_root=RAW_ROOT):
    path, approved_root = Path(path).resolve(), Path(approved_root).resolve()
    if not path.is_relative_to(approved_root):
        raise ValueError(f'Raw source outside approved root: {path}')
    if not path.is_file():
        raise FileNotFoundError(f'Raw source is not a file: {path}')
    return path

def validate_reuse_artifacts(artifacts, retained_ids, require_complete=True):
    """Validate exact folder/record/variant identities before cache reuse."""
    retained_ids = set(retained_ids)
    accepted, seen = [], set()
    for folder, name in artifacts:
        suffixes = REUSE_VARIANTS.get(folder)
        matches = [] if suffixes is None else [
            (recording, suffix) for recording in retained_ids for suffix in suffixes
            if name == recording + suffix]
        if len(matches) != 1:
            raise ValueError(f'Unrecognized reused artifact: {folder}/{name}')
        key = (folder, matches[0][0], matches[0][1])
        if key in seen:
            raise ValueError(f'Duplicate reused artifact identity: {folder}/{name}')
        seen.add(key); accepted.append((folder, name))
    expected = {(folder, recording, suffix) for folder, suffixes in REUSE_VARIANTS.items()
                for recording in retained_ids for suffix in suffixes}
    if require_complete and seen != expected:
        missing, extra = expected - seen, seen - expected
        raise ValueError(f'Incomplete reused artifact allowlist: missing={len(missing)}, extra={len(extra)}')
    return accepted

def select_reuse_candidates(data_root, retained_ids, require_complete=True):
    """Select only authorized P1/P2 files while ignoring well-formed P3 peers."""
    data_root = Path(data_root)
    retained_ids = set(retained_ids)
    legitimate_p3 = {f'WT_DIV{div}_P3_{well}' for div in range(14, 64, 7)
                     for well in ('A1', 'A2', 'A3', 'B1', 'B2', 'B3')}
    selected = []
    for folder, suffixes in REUSE_VARIANTS.items():
        expected_names = {recording + suffix for recording in retained_ids for suffix in suffixes}
        ignored_names = {recording + suffix for recording in legitimate_p3 for suffix in suffixes}
        for source in sorted((data_root / folder).iterdir()):
            if not source.is_file():
                continue
            if source.name in expected_names:
                selected.append((folder, source.name))
            elif source.name in ignored_names:
                continue
            elif re.match(r'^WT_DIV\d+_P[12]_', source.name):
                raise ValueError(f'Unauthorized or malformed P1/P2 artifact: {folder}/{source.name}')
            else:
                raise ValueError(f'Unrecognized artifact in reuse folder: {folder}/{source.name}')
    return validate_reuse_artifacts(selected, retained_ids, require_complete=require_complete)

def validate_duration_bins(records):
    required = {'recording', 'DIV', 'plate', 'duration_s'}
    if not required.issubset(records.columns):
        raise ValueError(f'Missing duration evidence columns: {sorted(required-set(records.columns))}')
    values = records.copy()
    values['bin_count_25ms'] = np.ceil(values.duration_s.astype(float) / .025).astype(int)
    short = values[values.bin_count_25ms != 12000]
    expected_short = (short.plate.astype(int).eq(3) & short.DIV.astype(int).isin([28, 63]) &
                      short.duration_s.astype(float).eq(299.75) & short.bin_count_25ms.eq(11990))
    if len(short) != 12 or not expected_short.all():
        raise ValueError('Expected exactly 12 P3 DIV28/DIV63 records with 11,990 bins')
    regular = values.drop(short.index)
    if not regular.duration_s.astype(float).eq(300.0).all() or not regular.bin_count_25ms.eq(12000).all():
        raise ValueError('All other retained records must have 300 s and 12,000 bins')
    distribution = values.bin_count_25ms.value_counts().sort_index()
    return {'short_recordings': sorted(short.recording.astype(str)),
            'short_duration_s': 299.75,
            'bin_count_distribution': {str(int(k)): int(v) for k, v in distribution.items()}}

def main():
    for directory in ['logs','tables','inputs','figures','report','manuscript','tests/legacy','data_intermediate/metadata']:
        (ROOT/directory).mkdir(parents=True, exist_ok=True)
    required_outputs = [ROOT/'tables/input_provenance.csv', ROOT/'inputs/source_copy_manifest.csv',
                        ROOT/'logs/bootstrap_summary.json',ROOT/'data_intermediate/metadata/recording_metadata.csv']
    if any(p.exists() for p in required_outputs):
        raise FileExistsError('Bootstrap summary already exists; verify/reuse rather than overwrite')
    copies = []
    for path in iter_original_code_files(OLD/'code'):
        copies.append(copy_verified(path, ROOT/'code'/path.relative_to(OLD/'code'), ROOT))
    for path in sorted((OLD/'tests').glob('*')):
        if path.is_file() and path.suffix in {'.py','.m'}:
            copies.append(copy_verified(path, ROOT/'tests/legacy'/path.name, ROOT))
    for name in ['audit_native_events.py','audit_sources.py']:
        copies.append(copy_verified(CCS/'code/vendor_audit'/name, ROOT/'code/provenance_vendor'/name, ROOT))
    for name in ['native_spk_headers.json','native_spk_all_well_event_inventory.csv','raw_file_inventory.csv']:
        copies.append(copy_verified(CCS/'audit'/name, ROOT/'inputs'/name, ROOT))
    copies.append(copy_verified(OLD/'analysis_config.yaml',ROOT/'inputs/original_analysis_config.yaml',ROOT))
    old_plan = ROOT.parent/'V2_Analysis/analysis_plan.md'
    copies.append(copy_verified(old_plan,ROOT/'inputs/approved_original_calibration_plan.md',ROOT))
    for path in sorted((ROOT.parent/'codex').glob('*.docx')):
        copies.append(copy_verified(path,ROOT/'inputs/manuscript_sources'/path.name,ROOT))
    sys.path.insert(0,str(ROOT/'code/provenance_vendor'))
    from audit_native_events import load_spk
    metadata = json.loads((ROOT/'inputs/native_spk_headers.json').read_text(encoding='utf8'))
    meta = [m for m in metadata if m['dataset']=='ccs_current_3_independent']
    if len(meta)!=24 or {(m['DIV'],m['plate']) for m in meta}!={(d,p) for d in range(14,64,7) for p in [1,2,3]}:
        raise RuntimeError('Corrected raw manifest must have 24 unique age/plate inputs')
    inventory = pd.read_csv(ROOT/'inputs/raw_file_inventory.csv').set_index('path')
    prior = pd.read_csv(ROOT/'inputs/native_spk_all_well_event_inventory.csv')
    prior = prior[prior.dataset=='ccs_current_3_independent'].set_index('recording')
    old_metadata = pd.read_csv(OLD/'data_intermediate/metadata/recording_metadata.csv').set_index('recording')
    rows, output_metadata, events_seen = [], [], set()
    for m in sorted(meta,key=lambda v:(v['DIV'],v['plate'])):
        raw_path = validate_raw_source(m['path'])
        rawhash=sha256(raw_path)
        if rawhash!=inventory.loc[m['path'],'sha256']:
            raise RuntimeError(f'Raw input changed since CCS verification: {m["path"]}')
        wells, _, native_ok = load_spk(m)
        for well, events in wells.items():
            rid=f'WT_DIV{m["DIV"]}_P{m["plate"]}_{well}'
            if events['sha256']!=prior.loc[rid,'spike_event_sha256']:
                raise RuntimeError(f'Raw event hash changed: {rid}')
            if events['sha256'] in events_seen:
                raise RuntimeError(f'Duplicate complete raw event array: {rid}')
            events_seen.add(events['sha256'])
            excluded=rid==EXCLUDED
            row=dict(recording=rid,DIV=m['DIV'],plate=m['plate'],well=well,excluded=excluded,
                     duration_s=m['duration_s'],fs=m['fs'],raw_path=m['path'],raw_sha256=rawhash,
                     event_sha256=events['sha256'],n_spikes=events['n'],native_reference_equal=native_ok)
            if not excluded:
                source=(CCS/'processed/spike_arrays' if m['plate']==3 else OLD/'data_intermediate/spike_arrays')/f'{rid}_arrays.mat'
                d=loadmat(source,squeeze_me=True,struct_as_record=False)
                if not events_equal(d['spike_time_s'],d['spike_channel_index'],events['times'],events['channels']):
                    raise RuntimeError(f'Cached events differ from newly read raw SPK: {rid}')
                if float(d['duration_s'])!=m['duration_s'] or float(d['fs'])!=m['fs']:
                    raise RuntimeError(f'Cached duration/sampling rate mismatch: {rid}')
                dest=ROOT/'data_intermediate/spike_arrays'/source.name
                copy=copy_verified(source,dest,ROOT);copies.append(copy)
                row.update(arrays_path=str(dest),arrays_sha256=copy['sha256'],exact_raw_array_match=True,
                           action='corrected_P3' if m['plate']==3 else 'preserved_P1P2')
                mr=old_metadata.loc[rid].to_dict()
                counts=np.bincount(np.asarray(d['spike_channel_index'],int).ravel(),minlength=65)[1:]
                mr.update(recording=rid,source_file=Path(m['path']).name,spk_path=m['path'],excluded=0,
                          n_spikes=events['n'],n_active_electrodes_0p01Hz=int(np.sum(counts/m['duration_s']>=.01)),
                          mean_firing_rate_hz_all=float(counts.mean()/m['duration_s']),duration_s_meta=m['duration_s'],
                          fs_meta=m['fs'],output_arrays_mat=str(dest))
                output_metadata.append(mr)
            rows.append(row)
        print(f'Raw/cache verified: DIV{m["DIV"]} P{m["plate"]}, six distinct wells',flush=True)
    if len(events_seen)!=144 or len(output_metadata)!=143:
        raise RuntimeError('Incorrect verified cohort size')
    # Reuse only record-local P1/P2 artifacts. No P3 or global combined tables are copied.
    reuse=[]
    retained_p1p2 = {r['recording'] for r in rows if not r['excluded'] and r['plate'] in (1, 2)}
    for folder, name in select_reuse_candidates(OLD/'data_intermediate', retained_p1p2):
        source = OLD/'data_intermediate'/folder/name
        copied=copy_verified(source,ROOT/'data_intermediate'/folder/name,ROOT)
        copies.append(copied);reuse.append(copied)
    pd.DataFrame(rows).to_csv(ROOT/'tables/input_provenance.csv',index=False,mode='x')
    pd.DataFrame(output_metadata).sort_values(['DIV','plate_index','well_label']).to_csv(required_outputs[3],index=False,mode='x')
    pd.DataFrame(copies).to_csv(ROOT/'inputs/source_copy_manifest.csv',index=False,mode='x')
    pd.DataFrame(reuse).to_csv(ROOT/'inputs/p1p2_local_artifact_reuse.csv',index=False,mode='x')
    duration_evidence = validate_duration_bins(pd.DataFrame(rows).query('not excluded'))
    summary=dict(raw_files_verified=24,raw_wells=144,distinct_full_raw_event_arrays=144,retained_records=143,
                 corrected_p3=48,preserved_p1p2=95,excluded_recording=EXCLUDED,
                 copied_source_files=len(copies),reused_p1p2_local_artifacts=len(reuse),
                 p3_short_recordings=len(duration_evidence['short_recordings']),
                 p3_short_duration_s=duration_evidence['short_duration_s'],
                 primary_25ms_bin_counts=duration_evidence['bin_count_distribution'],
                 source_writes=False,primary_recomputation_complete=False,calibration_complete=False)
    required_outputs[2].write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
