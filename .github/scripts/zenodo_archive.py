"""One-time publication of the exact, immutable manuscript release on Zenodo.

Run manually, with a short-lived token in an encrypted GitHub Actions secret.
Never resets a completed file, changes the scientific tag, or edits metadata.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.parse

RECORD = '23299960'
BASE = f'https://zenodo.org/api/records/{RECORD}/draft'
REPO = 'Jia-Jackson/developmental-triplet-coactivity'
COMMIT = '170c82d4bf76a49eef5f42165c490305c6abd7a3'


def validate_completed(entry, expected):
    if (entry.get('key') != expected['key'] or entry.get('status') != 'completed'
            or entry.get('size') != expected['size']
            or entry.get('checksum') != expected['md5']):
        raise ValueError('Completed file differs from verified release: ' + expected['key'])


def validate_resettable(entry, name):
    if (entry.get('key') != name or entry.get('status') != 'pending'
            or entry.get('size') not in (None, 0) or entry.get('checksum')):
        raise ValueError('Refusing to reset any stored or unrelated file')


def api(url, method='GET', value=None):
    if urllib.parse.urlparse(url).hostname != 'zenodo.org':
        raise ValueError('Unexpected API host')
    for attempt in range(5):
        response = requests.request(method, url, json=value,
                                    headers={'Authorization': 'Bearer ' + os.environ['ZENODO_ARCHIVE_TOKEN']},
                                    timeout=120)
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 4:
            print(f'Archive API HTTP {response.status_code}; bounded retry', flush=True)
            time.sleep(10)
            continue
        if not response.ok:
            raise RuntimeError(f'Archive API {method} returned HTTP {response.status_code}')
        return response.json() if response.content else None


def entries():
    return api(BASE + '/files')['entries']


def initialize(expected):
    name = expected['key']
    found = next((entry for entry in entries() if entry['key'] == name), None)
    if found and found['status'] == 'completed':
        validate_completed(found, expected)
        return found
    if found:
        validate_resettable(found, name)
        # Remove only our failed, empty upload placeholder in this draft.
        endpoint = BASE + '/files/' + urllib.parse.quote(name)
        try:
            api(endpoint, 'DELETE')
        except RuntimeError:
            if any(item['key'] == name for item in entries()):
                raise
    created = api(BASE + '/files', 'POST', [{'key': name}])
    candidates = created.get('entries', []) if isinstance(created, dict) else []
    if not candidates:
        candidates = entries()
    fresh = next(item for item in candidates if item['key'] == name)
    validate_resettable(fresh, name)
    return fresh


def transfer(expected, asset, path, entry):
    name = expected['key']
    if entry['status'] == 'completed':
        print('Verified existing archive file: ' + name, flush=True)
        return
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    with requests.get(asset['browser_download_url'], stream=True, timeout=(30, 120)) as response:
        if not response.ok:
            raise RuntimeError(f'Public release download HTTP {response.status_code}: {name}')
        with path.open('wb') as output:
            for block in response.iter_content(1024 * 1024):
                output.write(block)
                sha.update(block)
                md5.update(block)
    if (path.stat().st_size != expected['size'] or sha.hexdigest() != expected['sha256']
            or 'md5:' + md5.hexdigest() != expected['md5']):
        raise ValueError('Downloaded release bytes differ from local verified file: ' + name)
    print('Downloaded and checksum-verified: ' + name, flush=True)
    for attempt in range(3):
        endpoint = entry['links']['content']
        if (urllib.parse.urlparse(endpoint).hostname != 'zenodo.org'
                or not urllib.parse.urlparse(endpoint).path.startswith(f'/api/records/{RECORD}/draft/files/')):
            raise ValueError('Unexpected file transfer endpoint')
        # curl reads early HTTP rejection before sending the whole large file.
        # Its bearer credential is supplied only through an anonymous stdin pipe.
        quote = lambda text: '"' + str(text).replace('\\', '\\\\').replace('"', '\\"') + '"'
        configuration = '\n'.join([
            'url = ' + quote(endpoint), 'request = "PUT"', 'upload-file = ' + quote(path),
            'header = ' + quote('Authorization: Bearer ' + os.environ['ZENODO_ARCHIVE_TOKEN']),
            'header = "Content-Type: application/octet-stream"',
            'header = "Expect: 100-continue"', 'expect100-timeout = 3',
            'connect-timeout = 30', 'max-time = 5400', 'http1.1', 'silent', 'show-error',
            'write-out = "\\n%{http_code} %{size_upload} %{speed_upload}"',
        ]) + '\n'
        print('Starting Zenodo transfer: ' + name, flush=True)
        result = subprocess.run(['curl', '-q', '--config', '-'], input=configuration.encode(),
                                capture_output=True, timeout=5430)
        output = result.stdout.decode(errors='replace')
        transfer_summary = output.rsplit('\n', 1)[-1].split()
        status = int(transfer_summary[0]) if transfer_summary else 0
        detail = result.stderr.decode(errors='replace').replace(os.environ['ZENODO_ARCHIVE_TOKEN'], '[REDACTED]')
        if not result.returncode and status in (200, 201):
            if int(transfer_summary[1]) != expected['size']:
                raise ValueError('Transmitted-byte count mismatch')
            print(f'Transferred {name}: {float(transfer_summary[2])/1024:.1f} KiB/s', flush=True)
            committed = api(entry['links']['commit'], 'POST')
            validate_completed(committed, expected)
            print('Archived and checksum-verified: ' + name, flush=True)
            return
        current = next(item for item in entries() if item['key'] == name)
        if current['status'] == 'completed':
            validate_completed(current, expected)
            return
        if attempt == 2:
            raise RuntimeError(f'Archive transfer HTTP {status}, curl {result.returncode}: {name}; {detail[:200]}')
        print(f'Archive transfer HTTP {status}, curl {result.returncode}: {name}; {detail[:200]}', flush=True)
        body = output.rsplit('\n', 1)[0].replace(os.environ['ZENODO_ARCHIVE_TOKEN'], '[REDACTED]')
        if status >= 400:
            print('Archive response: ' + body[:300], flush=True)
        entry = initialize(expected)


def select_expected(manifest, name):
    matching = [item for item in manifest if item['key'] == name]
    if len(matching) != 1:
        raise ValueError('Only a verified manuscript file may be transferred')
    return matching[0]


def main(one_file=None, publish=False):
    global requests
    import requests
    manifest = json.loads(Path(__file__).with_name('zenodo_manifest.json').read_text())
    if len(manifest) != 10 or len({item['key'] for item in manifest}) != 10:
        raise ValueError('Expected exactly ten archive files')
    public_url = 'https://zenodo.org/api/records/' + RECORD
    public = requests.get(public_url, timeout=60)
    if public.ok:
        print('Record is already public; no archive mutation performed', flush=True)
        return
    release = requests.get(f'https://api.github.com/repos/{REPO}/releases/tags/v1.0.0', timeout=60).json()
    tag = requests.get(f'https://api.github.com/repos/{REPO}/git/ref/tags/v1.0.0', timeout=60).json()
    if release.get('draft') or tag.get('object', {}).get('sha') != COMMIT:
        raise ValueError('Scientific release changed')
    assets = {item['name']: item for item in release['assets']}
    if set(assets) != {item['key'] for item in manifest}:
        raise ValueError('Public release file set mismatch')
    for item in manifest:
        asset = assets[item['key']]
        if asset['size'] != item['size'] or asset.get('digest') != 'sha256:' + item['sha256']:
            raise ValueError('Public release digest mismatch')
    if any(item['key'] not in assets for item in entries()):
        raise ValueError('Unexpected archive entry; no files changed')
    if one_file:
        expected = select_expected(manifest, one_file)
        prepared = initialize(expected)
        with tempfile.TemporaryDirectory(prefix='verified-manuscript-archive-') as folder:
            transfer(expected, assets[one_file], Path(folder) / one_file, prepared)
        return
    if not publish:
        raise ValueError('Choose one verified file or the final publication check')
    completed = {item['key']: item for item in entries()}
    if set(completed) != set(assets):
        raise ValueError('Final archive set mismatch')
    for item in manifest:
        validate_completed(completed[item['key']], item)
    # Read metadata only after every file has stored size/checksum. This avoids
    # serialization problems for incomplete drafts; publication still checks it.
    record = api(BASE)
    metadata = record['metadata']
    if (record.get('is_published') or not metadata['title'].startswith('Developmental triplet coactivity')
            or len(metadata['creators']) != 6 or metadata.get('version') != '1.0.0'):
        raise ValueError('Unexpected manuscript metadata or publication state')
    published = api(BASE + '/actions/publish', 'POST')
    if not published.get('is_published'):
        raise RuntimeError('Publication was not confirmed by Zenodo')
    print('PUBLISHED: https://zenodo.org/records/' + RECORD, flush=True)
    print('DOI: https://doi.org/10.5281/zenodo.' + RECORD, flush=True)
    print('All ten archive files match their verified release bytes.', flush=True)


if __name__ == '__main__':
    arguments = argparse.ArgumentParser()
    modes = arguments.add_mutually_exclusive_group(required=True)
    modes.add_argument('--file')
    modes.add_argument('--publish', action='store_true')
    args = arguments.parse_args()
    main(args.file, args.publish)
