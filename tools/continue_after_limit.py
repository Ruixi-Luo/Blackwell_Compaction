"""本轮本地压缩实验达到尝试上限后，断网复制缓存到新运行，保留原记录。

Only run_name and max_api_attempts change. Existing responses, compression
outputs and the complete attempt ledger are copied byte-for-byte and verified.
"""
import argparse
import re
import shutil
import socket
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blackwell.client import parse_probability
from blackwell.storage import read_json, write_json, file_hash, fingerprint
from run_experiment import prepare_run

def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--new-run-name', required=True)
    parser.add_argument('--max-attempts', required=True, type=int)
    args = parser.parse_args()
    require(re.fullmatch(r'[A-Za-z0-9_-]+', args.new_run_name), 'Invalid run name.')
    config_path = ROOT / args.config
    require(config_path.resolve().is_relative_to(ROOT / 'configs'), 'Config must be in configs/.')
    old_config = read_json(config_path)
    old_run = ROOT / 'runs' / old_config['run_name']
    old_manifest = read_json(old_run / 'manifest.json')
    ledger = read_json(old_run / 'api_attempts.json')
    require(len(ledger) == old_config['max_api_attempts'], 'Original attempt cap has not been exhausted.')
    require(args.max_attempts > old_config['max_api_attempts'], 'Continuation cap must be larger.')
    require(old_manifest['identity']['config'] == old_config, 'Original config differs from its manifest.')
    new_config = {**old_config, 'run_name':args.new_run_name, 'max_api_attempts':args.max_attempts}
    new_run = ROOT / 'runs' / new_config['run_name']
    require(not new_run.exists(), 'Do not overwrite an existing continuation.')
    require({k for k in new_config if new_config[k] != old_config[k]} == {'run_name','max_api_attempts'}, 'Prediction settings must remain identical.')
    for name in ('api_attempts.json','input_preflight.json','local_compression_check.json'):
        require((old_run / name).is_file(), 'Only runs with local compression preflight are supported.')

    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Recovery must not call the network')):
        _, _, new_manifest = prepare_run(new_config, new_run, offline=True)
        for key in ('examples_sha256','dataset','local_compressors'):
            require(new_manifest['identity'][key] == old_manifest['identity'][key], 'Changed identity: '+key)
        require(new_manifest['identity']['code']['files_sha256'] == old_manifest['identity']['code']['files_sha256'], 'Core source code changed.')
        copied = {}
        for folder in ('api_cache','local_compression_cache'):
            (new_run / folder).mkdir()
            for source in sorted((old_run / folder).glob('*.json')):
                record = read_json(source)
                require(fingerprint(record['request']) == source.stem, 'Request hash differs.')
                if folder == 'api_cache':
                    choice = record['response']['choices'][0]
                    require(choice['finish_reason'] == 'stop', 'Truncated response cannot be reused.')
                    parse_probability(choice['message']['content'])
                else:
                    require(fingerprint(record['result']) == record['result_sha256'], 'Compression hash differs.')
                target = new_run / folder / source.name
                shutil.copyfile(source, target)
                require(file_hash(source) == file_hash(target), 'Copied cache differs.')
                copied[str(target.relative_to(new_run)).replace('\\','/')] = file_hash(target)
        for name in ('api_attempts.json','input_preflight.json','local_compression_check.json'):
            shutil.copyfile(old_run / name, new_run / name)
            require(file_hash(old_run / name) == file_hash(new_run / name), 'Copied record differs.')
            copied[name] = file_hash(new_run / name)
        receipt = {'reason':'HTTP 429 retries exhausted the original administrative cap before all questions completed.',
            'source_run':old_config['run_name'], 'source_config':old_config, 'continuation_config':new_config,
            'source_complete_questions':len(read_json(old_run/'predictions.json')),
            'copied_api_responses':len(list((new_run/'api_cache').glob('*.json'))),
            'copied_local_compressions':len(list((new_run/'local_compression_cache').glob('*.json'))),
            'prior_attempts_preserved':len(ledger), 'new_api_calls':0,
            'identical_examples_code_and_compressor_runtime':True, 'files_sha256':copied}
        write_json(new_run/'continuation_receipt.json', receipt)
        old_manifest.update(status='api_attempt_limit_reached', continuation_run=new_config['run_name'])
        write_json(old_run/'manifest.json', old_manifest)
        write_json(config_path, new_config)
    print({k:v for k,v in receipt.items() if k not in ('files_sha256','source_config','continuation_config')})


if __name__ == '__main__':
    main()
