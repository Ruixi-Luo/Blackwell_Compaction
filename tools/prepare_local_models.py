"""只下载固定版本的公开权重并验 SHA256；不调用 LLM API。"""
import argparse
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blackwell.compressors.runtime import configure_cache, file_digest, verify_assets
from blackwell.storage import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Verify existing files without downloading.')
    args = parser.parse_args()
    configure_cache()
    assets = read_json(ROOT / 'configs/local_models.lock.json')
    if not args.check_only:
        for method, spec in assets.items():
            folder = ROOT / spec['directory']
            folder.mkdir(parents=True, exist_ok=True)
            for name, info in spec['files'].items():
                target = folder / name
                if target.exists():
                    continue  # Final verification catches corrupt or changed files.
                partial = target.with_suffix(target.suffix + '.partial')
                print(f'Downloading {method}/{name} ({info["bytes"]:,} bytes)', flush=True)
                request = Request(info['url'], headers={'User-Agent': 'Blackwell-Compaction/0.2'})
                with urlopen(request, timeout=120) as response, partial.open('wb') as handle:
                    while block := response.read(4 * 1024 * 1024):
                        handle.write(block)
                if partial.stat().st_size != info['bytes'] or file_digest(partial) != info['sha256']:
                    raise ValueError(f'Asset checksum mismatch: {method}/{name}')
                os.replace(partial, target)
        # Upstream LLMLingua uses this encoding for its own length reporting.
        # Pre-cache it here on D: so actual compression stays offline.
        import tiktoken
        tiktoken.encoding_for_model('gpt-3.5-turbo')
    verify_assets(assets)
    print('All pinned model assets verified. No LLM API calls made.')


if __name__ == '__main__':
    main()
