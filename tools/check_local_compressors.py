"""在选定样本上只运行本地压缩，禁止任何网络；不预测、不读取 question/label。"""
import argparse
import math
import socket
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from run_experiment import load_config, prepare_run, selected_methods
from blackwell.compressors import LOCAL_METHODS
from blackwell.local_cache import LocalCompressionCache
from blackwell.storage import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/local_comparison.json')
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    run = ROOT / 'runs' / config['run_name']
    tokenizer, examples, manifest = prepare_run(config, run)
    cache = LocalCompressionCache(run, manifest['identity']['local_compressors'])
    records = []
    # 包括 Hugging Face、NLTK、tiktoken 的意外下载也会在这里失败。
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Compression must be offline')):
        for method in (m for m in selected_methods(config) if m in LOCAL_METHODS):
            for index, example in enumerate(examples, 1):
                context = example['context']
                original = tokenizer.count(context)
                budget = max(1, math.floor(config['retained_ratio'] * original))
                result = cache.compress(context, budget, method, config['retained_ratio'])
                count = tokenizer.count(result['text'])
                records.append({'id': example['id'], 'method': method, 'original_tokens': original,
                                'compressed_tokens': count, 'budget_tokens': budget,
                                'retained_ratio': count / original, 'cache_hash': result['local_compression_hash']})
                print(f'{method} [{index}/{len(examples)}]: {original} -> {count} tokens', flush=True)
    write_json(run / 'local_compression_check.json', {
        'status': 'passed', 'socket_connections_blocked': True, 'llm_api_calls': 0, 'records': records})


if __name__ == '__main__':
    main()
