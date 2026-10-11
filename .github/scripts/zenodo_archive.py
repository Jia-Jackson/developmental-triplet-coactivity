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
PART_SIZE = 8 * 1024 * 1024


def validate_completed(entry, expected):
    if (entry.get('key') != expected['key'] or entry.get('status') != 'completed'
            or entry.get('size') != expected['size']
            or entry.get('checksum') != expected['md5']):
        raise ValueError('Completed file differs from verified release: ' + expected['key'])


def validate_resettable(entry, name):
    if (entry.get('key') != name or entry.get('status') != 'pending'
            or entry.get('size') not in (None, 0) or entry.get('checksum')):
        raise ValueError('Refusing to reset any stored or unrelated file')


def chunk_ranges(size, part_size):
    if size <= 0 or part_size <= 0:
        raise ValueError('Positive file and part sizes are required')
    return [(number + 1, offset, min(part_size, size - offset))
            for number, offset in enumerate(range(0, size, part_size))]


def is_resumable_multipart(entry, expected, part_size):
    return (entry.get('key') == expected['key'] and entry.get('status') == 'pending'
            and entry.get('size') == expected['size']
            and entry.get('transfer', {}).get('type') == 'M'
            and entry.get('transfer', {}).get('part_size') == part_size
            and len(entry.get('links', {}).get('parts', [])) == len(chunk_ranges(expected['size'], part_size)))


def await_checksum(entry, expected):
    for attempt in range(120):
        if entry.get('checksum') == expected['md5']:
            validate_completed(entry, expected)
            return entry
        if (entry.get('status') != 'completed' or entry.get('size') != expected['size']
                or (entry.get('checksum') and not entry['checksum'].startswith('multipart:'))):
            raise ValueError('Completed archive file has an unexpected checksum or state')
        if attempt % 6 == 0:
            print('Waiting for Zenodo server-computed whole-file MD5: ' + expected['key'], flush=True)
        time.sleep(10)
        entry = next(item for item in entries() if item['key'] == expected['key'])
    raise RuntimeError('Server checksum computation did not complete within the bounded wait')


def api(url, method='GET', value=None):
    if urllib.parse.urlparse(url).hostname != 'zenodo.org':
        raise ValueError('Unexpected API host')
    for attempt in range(5):
        response = requests.request(method, url, json=value,
                                    headers={'Authorization': 'Bearer ' + os.environ['ZENODO_ARCHIVE_TOKEN'],
                                             'Accept': 'application/vnd.inveniordm.v1+json'},
                                    timeout=120)
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 4:
            print(f'Archive API {method} HTTP {response.status_code}; bounded retry', flush=True)
            time.sleep(15 * (attempt + 1) if response.status_code == 429 else 10)
            continue
        if not response.ok:
            raise RuntimeError(f'Archive API {method} returned HTTP {response.status_code}')
        return response.json() if response.content else None


def entries():
    return api(BASE + '/files')['entries']


def github_read(path):
    headers = {'User-Agent': 'verified-manuscript-archive'}
    token = os.environ.get('GH_ARCHIVE_VERIFICATION_TOKEN')
    if token:
        headers['Authorization'] = 'Bearer ' + token
    url = f'https://api.github.com/repos/{REPO}/' + path
    for attempt in range(4):
        response = requests.get(url, headers=headers, timeout=60)
        if response.status_code == 200:
            return response.json()
        if response.status_code in (403, 429, 500, 502, 503, 504) and attempt < 3:
            print(f'GitHub verification HTTP {response.status_code}; bounded retry', flush=True)
            time.sleep(20)
            continue
        raise RuntimeError(f'GitHub verification unavailable: HTTP {response.status_code}; scientific files were not changed')


def initialize(expected):
    name = expected['key']
    found = next((entry for entry in entries() if entry['key'] == name), None)
    if found and found['status'] == 'completed':
        return await_checksum(found, expected)
    if found and is_resumable_multipart(found, expected, PART_SIZE):
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
    ranges = chunk_ranges(expected['size'], PART_SIZE)
    created = api(BASE + '/files', 'POST', [{'key': name, 'size': expected['size'],
                                           'transfer': {'type': 'M', 'parts': len(ranges),
                                                        'part_size': PART_SIZE}}])
    candidates = created.get('entries', []) if isinstance(created, dict) else []
    if not candidates:
        candidates = entries()
    fresh = next(item for item in candidates if item['key'] == name)
    if not is_resumable_multipart(fresh, expected, PART_SIZE):
        raise ValueError('Multipart file did not initialize with the exact expected size and part count')
    return fresh


def multipart_transfer(expected, path, entry):
    parts = {int(part['part']): part['url'] for part in entry['links']['parts']}
    ranges = chunk_ranges(expected['size'], PART_SIZE)

    def upload_part(part):
        number, offset, length = part
        endpoint = parts[number]
        parsed = urllib.parse.urlparse(endpoint)
        host = parsed.hostname or ''
        if parsed.scheme != 'https' or not (host == 'zenodo.org' or host.endswith('.cern.ch')
                                              or host.endswith('.amazonaws.com')):
            raise ValueError('Unexpected multipart storage host')
        # Signed storage URLs are never printed; the Zenodo PAT is sent only
        # to Zenodo, never to a separate storage endpoint.
        partial = path.with_name(path.name + f'.transport-part-{number}')
        with path.open('rb') as source:
            source.seek(offset)
            payload = source.read(length)
        if len(payload) != length:
            raise ValueError('Multipart source range was truncated')
        partial.write_bytes(payload)
        quote = lambda value: '"' + str(value).replace('\\', '\\\\').replace('"', '\\"') + '"'
        config = ['url = ' + quote(endpoint), 'request = "PUT"', 'upload-file = ' + quote(partial),
                  'header = "Content-Type: application/octet-stream"',
                  'header = "Expect: 100-continue"', 'connect-timeout = 30', 'max-time = 600',
                  'expect100-timeout = 3', 'silent', 'show-error', 'http1.1',
                  'write-out = "\\n%{http_code} %{size_upload}"']
        if host == 'zenodo.org':
            config.append('header = ' + quote('Authorization: Bearer ' + os.environ['ZENODO_ARCHIVE_TOKEN']))
        for attempt in range(4):
            response = subprocess.run(['curl', '-q', '--config', '-'],
                                      input=('\n'.join(config) + '\n').encode(),
                                      capture_output=True, timeout=630)
            summary = response.stdout.decode(errors='replace').rsplit('\n', 1)[-1].split()
            status = int(summary[0]) if summary else 0
            if not response.returncode and status in (200, 201, 204) and int(summary[1]) == length:
                partial.unlink()
                return number
            print(f'Retrying only transport part {number} of {expected["key"]}: HTTP {status}, curl {response.returncode}', flush=True)
            if attempt == 3:
                raise RuntimeError('Multipart transport failed; no signed URL or credential displayed')
            time.sleep(20 * (attempt + 1) if status == 429 else 8)

    print(f'Multipart upload {expected["key"]}: {len(ranges)} transport parts', flush=True)
    count = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(upload_part, part) for part in ranges]
        for future in as_completed(futures):
            future.result()
            count += 1
            if count % 4 == 0 or count == len(ranges):
                print(f'Transport parts acknowledged {expected["key"]}: {count}/{len(ranges)}', flush=True)
    committed = api(entry['links']['commit'], 'POST')
    await_checksum(committed, expected)
    print('Archived and server-MD5-verified: ' + expected['key'], flush=True)


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
    if entry.get('transfer', {}).get('type') == 'M':
        multipart_transfer(expected, path, entry)
        return
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
    release = github_read('releases/tags/v1.0.0')
    tag = github_read('git/ref/tags/v1.0.0')
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
