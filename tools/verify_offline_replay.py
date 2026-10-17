"""断网并禁止加载本地模型，验证缓存重放逐字节复现预测/指标/账本。"""
import argparse
import os
import socket
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import run_experiment
from blackwell.storage import file_hash, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/local_comparison.json')
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    run = ROOT / 'runs' / config['run_name']
    names = ['predictions.json', 'metrics.json', 'api_attempts.json']
    before = {name: file_hash(run / name) for name in names}
    for name in ('SILICONFLOW_API_KEY', 'OPENAI_API_KEY'):
        os.environ.pop(name, None)
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')), \
         patch('blackwell.local_cache.create_compressor', side_effect=AssertionError('Models must not reload')), \
         patch.object(run_experiment, 'load_local_keys', return_value=None), \
         patch.object(sys, 'argv', ['run_experiment.py', '--config', args.config, '--offline']):
        run_experiment.main()
    after = {name: file_hash(run / name) for name in names}
    if before != after:
        raise ValueError('Replay changed predictions, metrics or API ledger.')
    execution = read_json(run / 'manifest.json')['last_execution']
    if execution['new_api_attempts'] != 0:
        raise ValueError('Unexpected API attempt during offline replay.')
    result = {'status': 'passed', 'api_keys_removed': True, 'socket_connections_blocked': True,
              'local_model_loading_blocked': True, 'new_api_attempts': 0,
              'cache_hits': execution['cache_hits'], 'byte_identical_files_sha256': after}
    write_json(run / 'replay_verification.json', result)
    print(result)


if __name__ == '__main__':
    main()
