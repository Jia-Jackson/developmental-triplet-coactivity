"""Restore original release ZIPs from this Zenodo record's portable volumes.

Uses Python's standard library only. Never overwrites an existing different
file. Verifies every volume MD5 and every restored original ZIP SHA256.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile


def restore(folder):
    folder = Path(folder).resolve()
    data = json.loads((folder / 'archive_manifest.json').read_text(encoding='utf-8'))
    output = folder / 'restored_archives'
    output.mkdir(exist_ok=True)
    for original in data['original_archives']:
        name = original['key']
        if Path(name).name != name:
            raise ValueError('Unsafe archive name')
        pieces = sorted((item for item in data['archive_files'] if item.get('original_key') == name),
                        key=lambda item: item['offset'])
        if not pieces:
            existing = folder / name
            if not existing.exists():
                print('Direct ZIP not downloaded; no restoration needed: ' + name)
                continue
            if hashlib.sha256(existing.read_bytes()).hexdigest() != original['sha256']:
                raise ValueError('Complete original ZIP checksum mismatch: ' + name)
            print('Verified complete ZIP: ' + name)
            continue
        destination = output / name
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != original['sha256']:
                raise ValueError('Refusing to overwrite a different existing file: ' + name)
            print('Verified previously restored ZIP: ' + name)
            continue
        sha = hashlib.sha256()
        offset = 0
        with tempfile.TemporaryDirectory(prefix='verified-zip-', dir=output) as temporary:
            staged = Path(temporary) / name
            with staged.open('wb') as target:
                for item in pieces:
                    if Path(item['key']).name != item['key'] or item['offset'] != offset:
                        raise ValueError('Unsafe name, missing or unordered portable volume')
                    payload = (folder / item['key']).read_bytes()
                    if (len(payload) != item['size'] or
                            'md5:' + hashlib.md5(payload).hexdigest() != item['md5']):
                        raise ValueError('Portable volume checksum mismatch: ' + item['key'])
                    target.write(payload)
                    sha.update(payload)
                    offset += len(payload)
            if offset != original['size'] or sha.hexdigest() != original['sha256']:
                raise ValueError('Restored original ZIP checksum mismatch: ' + name)
            if destination.exists():
                raise ValueError('Destination appeared; refusing to overwrite it')
            staged.rename(destination)
        print('Restored byte-identical original ZIP: ' + str(destination))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', default=str(Path(__file__).resolve().parent))
    restore(parser.parse_args().folder)
