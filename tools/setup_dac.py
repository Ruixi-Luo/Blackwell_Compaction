"""Fetch only the DAC files already pinned in configs/dac.lock.json.

Use --offline to verify the local installation without network or model calls.
The script never updates the pinned revision and stores everything under data/.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Only verify existing assets; never download.')
    args = parser.parse_args()
    lock = json.loads((ROOT/'configs/dac.lock.json').read_text(encoding='utf-8'))
    folder = (ROOT/lock['directory']).resolve()
    for name, record in lock['files'].items():
        target = (folder/name).resolve()
        if not target.is_relative_to(folder):
            raise ValueError('Asset path escapes the model directory.')
        if not target.exists():
            if args.offline:
                raise FileNotFoundError(f'Missing pinned asset: {name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix+'.download')
            with urlopen(record['url'], timeout=180) as response, temporary.open('wb') as output:
                while chunk := response.read(1024*1024):
                    output.write(chunk)
            if temporary.stat().st_size != record['bytes'] or digest(temporary) != record['sha256']:
                raise ValueError(f'Download checksum mismatch: {name}')
            temporary.replace(target)
        if target.stat().st_size != record['bytes'] or digest(target) != record['sha256']:
            raise ValueError(f'Installed asset checksum mismatch: {name}')
    print(f'Verified {len(lock["files"])} pinned DAC assets; no inference API was called.')


if __name__ == '__main__':
    main()
