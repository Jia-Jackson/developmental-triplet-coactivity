"""Archive immutable ZIP bytes as portable standard-upload volumes.

Zenodo's multipart route denies part uploads (upstream issue 1443). This uses
only its ordinary owner-authorized local file upload API, not multipart routes.
The original ZIP files, release tag and scientific contents remain unchanged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.parse
import zenodo_archive as archive

BASE = archive.BASE


def part_spec(name, size, part_size):
    if size <= 0 or part_size <= 0:
        raise ValueError('Positive sizes required')
    return [(f'{name}.part{number + 1:02}', offset, min(part_size, size-offset))
            for number, offset in enumerate(range(0, size, part_size))]


def validate_unwritten_multipart(entry, expected):
    if (entry.get('key') != expected['key'] or entry.get('status') != 'pending'
            or entry.get('size') != expected['size'] or entry.get('checksum')
            or entry.get('transfer', {}).get('type') != 'M'):
        raise ValueError('Refusing to remove any completed, unrelated or stored file')


def validate_complete(entry, expected):
    archive.validate_completed(entry, expected)


def is_empty_owned_standard(entry, name):
    return (entry.get('key') == name and entry.get('status') == 'pending'
            and entry.get('size') in (None, 0) and not entry.get('checksum')
            and entry.get('transfer', {}).get('type') == 'L')


def manifest():
    return json.loads(Path(__file__).with_name('zenodo_portable_manifest.json').read_text())


def reset_unwritten_multipart():
    originals = json.loads(Path(__file__).with_name('zenodo_manifest.json').read_text())
    preserved = {'raw_spk_DIV14.zip', 'raw_spk_DIV49.zip', 'repository_upload.zip'}
    for item in originals:
        if item['key'] in preserved:
            found = next(entry for entry in archive.entries() if entry['key'] == item['key'])
            validate_complete(found, item)
            continue
        found = next((entry for entry in archive.entries() if entry['key'] == item['key']), None)
        if found:
            validate_unwritten_multipart(found, item)
            archive.api(BASE + '/files/' + urllib.parse.quote(item['key']), 'DELETE')
            print('Removed only untransferred multipart placeholder: ' + item['key'], flush=True)


def standard_upload(expected, payload):
    digest = 'md5:' + hashlib.md5(payload).hexdigest()
    if len(payload) != expected['size'] or digest != expected['md5']:
        raise ValueError('Local portable-volume bytes differ')
    found = next((entry for entry in archive.entries() if entry['key'] == expected['key']), None)
    if found and found['status'] == 'completed':
        validate_complete(found, expected)
        return
    if found is None:
        initialized = archive.api(BASE + '/files', 'POST', [{'key': expected['key']}])
        found = next(entry for entry in initialized['entries'] if entry['key'] == expected['key'])
    if found.get('transfer', {}).get('type') != 'L' or found.get('status') != 'pending':
        raise ValueError('Standard upload entry has an unexpected transfer state')
    endpoint = found['links']['content']
    if (urllib.parse.urlparse(endpoint).hostname != 'zenodo.org'
            or not urllib.parse.urlparse(endpoint).path.startswith('/api/records/23299960/draft/files/')):
        raise ValueError('Unexpected ordinary upload endpoint')
    quote = lambda value: '"' + str(value).replace('\\', '\\\\').replace('"', '\\"') + '"'
    with tempfile.TemporaryDirectory(prefix='portable-manuscript-volume-') as folder:
        path = Path(folder) / expected['key']
        path.write_bytes(payload)
        config = '\n'.join(['url = ' + quote(endpoint), 'request = "PUT"',
            'upload-file = ' + quote(path),
            'header = ' + quote('Authorization: Bearer ' + os.environ['ZENODO_ARCHIVE_TOKEN']),
            'header = "Content-Type: application/octet-stream"', 'header = "Expect: 100-continue"',
            'expect100-timeout = 3', 'connect-timeout = 30', 'max-time = 900',
            'http1.1', 'silent', 'show-error', 'write-out = "\\n%{http_code} %{size_upload}"']) + '\n'
        for attempt in range(4):
            import time
            result = subprocess.run(['curl', '-q', '--config', '-'], input=config.encode(),
                                    capture_output=True, timeout=930)
            summary = result.stdout.decode(errors='replace').rsplit('\n', 1)[-1].split()
            status = int(summary[0]) if summary else 0
            if result.returncode == 0 and status in (200, 201):
                if int(summary[1]) != expected['size']:
                    raise ValueError('Ordinary transmitted byte count mismatch')
                committed = archive.api(found['links']['commit'], 'POST')
                validate_complete(committed, expected)
                print('Server MD5 verified: ' + expected['key'], flush=True)
                return
            if status in (401, 403):
                raise RuntimeError(f'Ordinary upload refused HTTP {status}; no permission workaround attempted')
            current = next(entry for entry in archive.entries() if entry['key'] == expected['key'])
            if current['status'] == 'completed':
                validate_complete(current, expected)
                return
            if current.get('checksum') == expected['md5'] and current.get('size') == expected['size']:
                validate_complete(archive.api(current['links']['commit'], 'POST'), expected)
                return
            if status == 400 and is_empty_owned_standard(current, expected['key']):
                # Only this worker's failed empty ordinary-upload entry can
                # retain a server-side upload lock after a gateway timeout.
                archive.api(BASE + '/files/' + urllib.parse.quote(expected['key']), 'DELETE')
                initialized = archive.api(BASE + '/files', 'POST', [{'key': expected['key']}])
                found = next(entry for entry in initialized['entries'] if entry['key'] == expected['key'])
                print('Reset only failed empty owned volume: ' + expected['key'], flush=True)
            print(f'Ordinary upload retry {expected["key"]}: HTTP {status}, curl {result.returncode}', flush=True)
            if attempt == 3:
                raise RuntimeError('Bounded standard upload retries exhausted')
            time.sleep(20)


def upload_original(name):
    import requests
    archive.requests = requests
    originals = json.loads(Path(__file__).with_name('zenodo_manifest.json').read_text())
    original = archive.select_expected(originals, name)
    expected_parts = [entry for entry in manifest() if entry.get('original_key') == name]
    if not expected_parts:
        found = next(entry for entry in archive.entries() if entry['key'] == name)
        validate_complete(found, original)
        print('Verified existing complete archive: ' + name, flush=True)
        return
    present = {entry['key']: entry for entry in archive.entries()}
    missing = []
    for item in expected_parts:
        existing = present.get(item['key'])
        if existing and existing['status'] == 'completed':
            validate_complete(existing, item)
        else:
            missing.append(item)
    if not missing:
        print('All original ZIP byte volumes already verified: ' + name, flush=True)
        return
    release = archive.github_read('releases/tags/v1.0.0')
    tag = archive.github_read('git/ref/tags/v1.0.0')
    if release.get('draft') or tag['object']['sha'] != archive.COMMIT:
        raise ValueError('Immutable scientific source mismatch')
    asset = next(asset for asset in release['assets'] if asset['name'] == name)
    if asset['size'] != original['size'] or asset['digest'] != 'sha256:' + original['sha256']:
        raise ValueError('Public source asset digest mismatch')
    with requests.get(asset['browser_download_url'], timeout=(30, 120)) as response:
        response.raise_for_status()
        payload = response.content
    if (len(payload) != original['size'] or hashlib.sha256(payload).hexdigest() != original['sha256']
            or 'md5:' + hashlib.md5(payload).hexdigest() != original['md5']):
        raise ValueError('Public release bytes mismatch')
    print('Verified original ZIP bytes; uploading independent standard volumes: ' + name, flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(standard_upload, item,
                   payload[item['offset']:item['offset'] + item['size']]) for item in missing]
        for future in as_completed(futures):
            future.result()


def upload_volume(name):
    import requests
    archive.requests = requests
    expected = archive.select_expected(manifest(), name)
    if not expected.get('original_key'):
        raise ValueError('Only a known original-ZIP byte volume can be independently resumed')
    present = next((item for item in archive.entries() if item['key'] == name), None)
    if present and present['status'] == 'completed':
        validate_complete(present, expected)
        print('Verified existing portable volume: ' + name, flush=True)
        return
    originals = json.loads(Path(__file__).with_name('zenodo_manifest.json').read_text())
    original = archive.select_expected(originals, expected['original_key'])
    release = archive.github_read('releases/tags/v1.0.0')
    tag = archive.github_read('git/ref/tags/v1.0.0')
    if release.get('draft') or tag['object']['sha'] != archive.COMMIT:
        raise ValueError('Scientific source changed')
    asset = next(item for item in release['assets'] if item['name'] == original['key'])
    if asset['size'] != original['size'] or asset['digest'] != 'sha256:' + original['sha256']:
        raise ValueError('Public original-ZIP digest mismatch')
    start, size = expected['offset'], expected['size']
    end = start + size - 1
    with requests.get(asset['browser_download_url'], headers={'Range':f'bytes={start}-{end}'},
                      timeout=(30,120)) as response:
        response.raise_for_status()
        payload = response.content
        if response.status_code == 206:
            if response.headers.get('Content-Range') != f'bytes {start}-{end}/{original["size"]}':
                raise ValueError('Public byte-range response mismatch')
        elif response.status_code == 200:
            if len(payload) != original['size'] or hashlib.sha256(payload).hexdigest() != original['sha256']:
                raise ValueError('Whole-file fallback does not match scientific release')
            payload = payload[start:start+size]
        else:
            raise ValueError('Unexpected public source response')
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != expected['sha256']:
        raise ValueError('Independent portable-volume source bytes mismatch')
    print('Verified independent original-ZIP byte range: ' + name, flush=True)
    standard_upload(expected, payload)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--original')
    mode.add_argument('--item')
    args = parser.parse_args()
    if args.original:
        upload_original(args.original)
    elif args.item.endswith('.zip'):
        upload_original(args.item)
    else:
        upload_volume(args.item)
