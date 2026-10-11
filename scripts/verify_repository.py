import csv, gzip, hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for line in (root/'checksums.sha256').read_text(encoding='utf-8').splitlines():
    expected, relative = line.split('  ',1)
    h=hashlib.sha256()
    with (root/relative).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024),b''): h.update(block)
    if h.hexdigest()!=expected: raise RuntimeError('Checksum mismatch: '+relative)
with (root/'metadata/recordings.csv').open(encoding='utf-8',newline='') as handle:
    rows=list(csv.DictReader(handle))
retained=[r for r in rows if r['excluded']=='false']
assert len(rows)==144 and len(retained)==143
assert len({r['recording'] for r in rows})==144
for row in retained:
    with gzip.open(root/row['event_file'],'rt',encoding='utf-8',newline='') as handle:
        reader=csv.reader(handle)
        assert next(reader)==['spike_time_s','electrode_index_1based']
        count=0
        for time,electrode in reader:
            assert 0<=float(time)<=float(row['duration_s'])
            assert 1<=int(electrode)<=64
            count+=1
        assert count==int(row['n_spikes']),row['recording']
oversized=[p for p in root.rglob('*') if p.is_file() and '.git' not in p.parts and p.stat().st_size>100*1024**2]
assert not oversized,oversized
print('PASS: repository hashes; 144 candidate / 143 retained recordings; all event counts and indices')
