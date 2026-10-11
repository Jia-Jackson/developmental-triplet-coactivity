"""Refresh the file manifest after deliberate public metadata changes."""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
lines = []
ignored = {'.git', '__pycache__', '.pytest_cache', '.venv', 'reproduced', 'raw_spk'}
for path in sorted(root.rglob('*')):
    if not path.is_file() or path.name == 'checksums.sha256':
        continue
    if any(part in ignored for part in path.relative_to(root).parts):
        continue
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    lines.append(digest.hexdigest() + '  ' + path.relative_to(root).as_posix())
(root / 'checksums.sha256').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('Updated checksums for', len(lines), 'repository files')
