"""按数据集串行执行：离线压缩 → API 预测 → 审计 → 断网重放，避免共享限流。"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blackwell.storage import read_json, write_json
from blackwell.data import load_dataset

DEFAULT_CONFIGS = ['configs/hotpotqa_full.json', 'configs/qasper_train.json', 'configs/longbench_binary.json']


def run_tool(script, config, log_path):
    with log_path.open('w', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, '-u', script, '--config', config], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{script} failed; see {log_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS)
    configs = parser.parse_args().configs
    errors = []
    status = {'status': 'running', 'configs': configs, 'datasets': {}}
    status_path = ROOT / '.cache/suite_status.json'
    for config_path in configs:
        config = read_json(ROOT / config_path)
        name = Path(config_path).stem
        run = ROOT / 'runs' / config['run_name']
        status['datasets'][name] = {'run_name': config['run_name'], 'stage': 'local_compression'}
        write_json(status_path, status)
        try:
            check = ROOT / '.cache/preflight' / f'{name}.json'
            if not check.exists() or read_json(check)['status'] != 'passed':
                raise RuntimeError('Run input preflight before starting the suite.')
            checked = read_json(check)
            examples, _ = load_dataset(config, offline=True)
            if (checked['questions'] != len(examples) or
                    {r['context_group'] for r in checked['records']} != {e['context_group'] for e in examples}):
                raise RuntimeError('Preflight belongs to different inputs; rerun it for this configuration.')
            # check_local_compressors 禁止网络，并且只向压缩器传 context。
            # GPU 一次只保留这一组模型，API 也只运行一个数据集。
            print(f'{name}: local compression started', flush=True)
            run_tool('tools/check_local_compressors.py', config_path, ROOT / '.cache' / f'{name}_compression.log')
            shutil.copyfile(check, run / 'input_preflight.json')
            status['datasets'][name].update(stage='predictions')
            write_json(status_path, status)
            print(f'{name}: compression complete; API predictions started', flush=True)
            run_tool('run_experiment.py', config_path, ROOT / '.cache' / f'{name}_predictions.log')
            run_tool('tools/audit_run.py', config_path, ROOT / '.cache' / f'{name}_audit.log')
            run_tool('tools/verify_offline_replay.py', config_path, ROOT / '.cache' / f'{name}_replay.log')
            status['datasets'][name]['stage'] = 'complete_and_audited'
            print(f'{name}: predictions, audit and offline replay complete', flush=True)
        except Exception as error:
            status['datasets'][name].update(stage='failed', error=str(error))
            errors.append(str(error))
        write_json(status_path, status)
    status['status'] = 'failed' if errors else 'complete'
    write_json(status_path, status)
    if errors:
        raise RuntimeError('\n'.join(errors))


if __name__ == '__main__':
    main()
