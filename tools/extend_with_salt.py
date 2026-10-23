"""只新增 SALT-summary：冻结旧预测 → 离线压缩 → 新预测 → 配对评估/审计。

旧运行只读；新目录的 API 账本只记录新增调用。小样本固定取旧顺序前 N 题。
示例：python tools/extend_with_salt.py --config configs/salt_smoke.json --stage compress
然后依次运行 --stage predict、audit、replay。重启相同命令会复用缓存。
"""
import argparse
import math
import re
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blackwell.client import CachedChatClient, parse_probability
from blackwell.compression import Compactor
from blackwell.data import load_dataset
from blackwell.local_cache import LocalCompressionCache
from blackwell.prompts import prediction_messages
from blackwell.storage import file_hash, fingerprint, read_json, write_json
from run_experiment import code_provenance, load_config, load_local_keys, write_outputs

METHOD = 'salt_summary'
SOURCE_FILES = ('manifest.json', 'examples.json', 'predictions.json', 'metrics.json',
                'audit.json', 'replay_verification.json', 'api_attempts.json')
READER_FILES = ('blackwell/client.py', 'blackwell/prompts.py', 'blackwell/tokenization.py',
                'blackwell/metrics.py', 'blackwell/data.py', 'run_experiment.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_config(old, new):
    """只容许追加方法、固定前缀小样本、以及必要的运行/限流设置。"""
    allowed = {'run_name', 'source_run', 'methods', 'sample_size',
               'max_api_attempts', 'min_interval_seconds'}
    require({k: v for k, v in old.items() if k not in allowed} ==
            {k: v for k, v in new.items() if k not in allowed},
            'Reader, dataset, prompts and evaluation settings must stay unchanged.')
    require(new['methods'] == old['methods'] + [METHOD], 'Only append salt_summary to the old methods.')
    require(1 <= new['sample_size'] <= old['sample_size'], 'Invalid frozen-prefix sample size.')
    require(new['run_name'] != new['source_run'], 'The source run must remain read-only.')
    require(bool(re.fullmatch(r'[A-Za-z0-9_-]+', new['source_run'])), 'Invalid source run name.')


def read_source(config):
    require(bool(re.fullmatch(r'[A-Za-z0-9_-]+', config['source_run'])), 'Invalid source run name.')
    source = ROOT / 'runs' / config['source_run']
    manifest = read_json(source / 'manifest.json')
    old = manifest['identity']['config']
    check_config(old, config)
    require(manifest['status'] == 'complete', 'Only complete source runs can be reused.')
    for name in ('audit.json', 'replay_verification.json'):
        require(read_json(source / name)['status'] == 'passed', 'Source audit/replay must pass.')
    # 新增压缩器允许改变其代码，但读取概率、提示词、tokenizer、指标必须保持原实现。
    recorded = manifest['identity']['code']['files_sha256']
    for name in (*READER_FILES, *sorted(n for n in recorded if n.startswith('blackwell/datasets/'))):
        require(file_hash(ROOT / name) == recorded[name], f'Reused reader/data code changed: {name}')
    examples = read_json(source / 'examples.json')
    require(fingerprint(examples) == manifest['identity']['examples_sha256'], 'Source examples changed.')
    reloaded, provenance = load_dataset(old, offline=True)
    require(examples == reloaded and provenance == manifest['identity']['dataset'],
            'Source examples differ from pinned raw data and original selection.')
    rows = read_json(source / 'predictions.json')
    require(len(rows) == len(examples) == old['sample_size'], 'Source sample count differs.')
    require(len({e['id'] for e in examples}) == len(examples), 'Duplicate source IDs.')
    for row, example in zip(rows, examples):
        require(all(row[k] == v for k, v in example.items()), 'Source rows differ from examples.')
        require(list(row['conditions']) == old['methods'], 'Source conditions differ.')
    receipt = {'source_run': config['source_run'], 'old_methods': old['methods'], 'new_method': METHOD,
               'source_files_sha256': {n: file_hash(source / n) for n in SOURCE_FILES},
               'selection': 'first N rows of frozen source order', 'selected_samples': config['sample_size'],
               'source_samples': len(rows), 'source_reader_files_sha256': {n: recorded[n] for n in READER_FILES}}
    return source, manifest, examples[:config['sample_size']], rows[:config['sample_size']], receipt


def prepare(config):
    from blackwell.compressors.runtime import configure_cache, runtime_identity, verify_assets
    from blackwell.tokenization import ContextTokenizer, TOKENIZER_REVISION, TOKENIZER_SHA256
    source, old_manifest, examples, old_rows, receipt = read_source(config)
    configure_cache()
    verify_assets([METHOD])
    tokenizer = ContextTokenizer()
    code = code_provenance()
    code['files_sha256']['tools/extend_with_salt.py'] = file_hash(Path(__file__))
    identity = {'config': config, 'dataset': old_manifest['identity']['dataset'],
                'examples_sha256': fingerprint(examples), 'tokenizer_revision': TOKENIZER_REVISION,
                'tokenizer_sha256': TOKENIZER_SHA256, 'code': code,
                'local_compressors': runtime_identity([METHOD], config['compression_device']),
                'extension': receipt}
    run = ROOT / 'runs' / config['run_name']
    if (run / 'manifest.json').exists():
        manifest = read_json(run / 'manifest.json')
        saved = manifest['identity']
        require({**saved, 'code': saved['code']['files_sha256']} ==
                {**identity, 'code': identity['code']['files_sha256']},
                'Run identity changed; choose a new run_name.')
        require(read_json(run / 'examples.json') == examples, 'Saved extension examples changed.')
    else:
        manifest = {'identity': identity, 'status': 'prepared',
                    'created_utc': datetime.now(timezone.utc).isoformat()}
        write_json(run / 'manifest.json', manifest)
        write_json(run / 'examples.json', examples)
        write_json(run / 'extension_receipt.json', receipt)
        write_json(run / 'api_attempts.json', [])
        if (source / 'input_preflight.json').exists() and len(old_rows) == receipt['source_samples']:
            shutil.copyfile(source / 'input_preflight.json', run / 'input_preflight.json')
    return run, tokenizer, old_rows, manifest


def compact_only(config, run, tokenizer, old_rows, manifest):
    cache = LocalCompressionCache(run, manifest['identity']['local_compressors'])
    records = []
    # 传给压缩器的只有上下文和长度参数；网络封锁也覆盖模型的意外下载。
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Compression must be offline')):
        for i, row in enumerate(old_rows, 1):
            context = row['context']
            original = tokenizer.count(context)
            budget = max(1, math.floor(config['retained_ratio'] * original))
            result = cache.compress(context, budget, METHOD, config['retained_ratio'])
            count = tokenizer.count(result['text'])
            records.append({'id': row['id'], 'method': METHOD, 'original_tokens': original,
                            'compressed_tokens': count, 'budget_tokens': budget,
                            'retained_ratio': count / original, 'cache_hash': result['local_compression_hash']})
            print(f'Compress [{i}/{len(old_rows)}]: {original} -> {count} tokens', flush=True)
    write_json(run / 'local_compression_check.json', {'status': 'passed',
               'socket_connections_blocked': True, 'llm_api_calls': 0, 'records': records})


def predict_rows(config, run, tokenizer, old_rows, manifest, offline=False):
    require(read_json(run / 'local_compression_check.json')['status'] == 'passed', 'Run compression first.')
    client = CachedChatClient(config, run, offline)
    # 压缩阶段必须已完成；预测阶段禁止创建任何压缩模型。
    cache = LocalCompressionCache(run, manifest['identity']['local_compressors'], offline=True)
    compactor = Compactor(client, tokenizer, cache, config['retained_ratio'])
    rows = []
    for i, old_row in enumerate(old_rows, 1):
        context = old_row['context']
        original = tokenizer.count(context)
        require(original == old_row['original_tokens'], 'Original token count changed.')
        budget = max(1, math.floor(config['retained_ratio'] * original))
        compressed = compactor.compress(context, budget, METHOD)
        raw, request_hash = client.complete(prediction_messages(compressed['text'], old_row['question']),
                                            config['prediction_max_tokens'], json_mode=True)
        p_yes = parse_probability(raw)
        condition = {**compressed, 'p_yes': p_yes, 'predicted_label': int(p_yes >= .5),
                     'prediction_request_hash': request_hash,
                     'retained_ratio_actual': compressed['tokens'] / original}
        rows.append({**old_row, 'conditions': {**old_row['conditions'], METHOD: condition}})
        write_json(run / 'predictions.json', rows)
        print(f'Predict [{i}/{len(old_rows)}] {old_row["id"]}: salt_summary={p_yes:.3f}', flush=True)
    return rows, {'new_api_attempts': client.new_attempts, 'cache_hits': client.cache_hits,
                  'reused_forecasts': len(old_rows) * (len(config['methods']) - 1)}


def check_cached_prediction(folder, config, row, condition):
    """用完整请求体重建指纹；不只检查 model 或 messages 这两个字段。"""
    payload = {'model': config['model'], 'messages': prediction_messages(condition['text'], row['question']),
               'temperature': config['temperature'], 'max_tokens': config['prediction_max_tokens'],
               'enable_thinking': config['enable_thinking'], 'stream': False,
               'response_format': {'type': 'json_object'}}
    request = {'base_url': config['base_url'], 'payload': payload}
    key = condition['prediction_request_hash']
    require(fingerprint(request) == key, 'Prediction request hash differs from complete expected request.')
    cached = read_json(folder / 'api_cache' / f'{key}.json')
    require(cached['request'] == request, 'Cached prediction request differs.')
    raw, _ = CachedChatClient._extract(cached['response'], key)
    require(parse_probability(raw) == condition['p_yes'], 'Probability differs from raw response.')
    return key


def verify_old_conditions(rows, old_rows, old_methods):
    require(len(rows) == len(old_rows), 'Extension is incomplete.')
    count = 0
    for row, old in zip(rows, old_rows):
        require({k: v for k, v in row.items() if k != 'conditions'} ==
                {k: v for k, v in old.items() if k != 'conditions'}, 'Sample/order changed.')
        require(list(row['conditions']) == old_methods + [METHOD], 'Unpaired conditions.')
        for method in old_methods:
            require(row['conditions'][method] == old['conditions'][method], 'Reused condition changed.')
            count += 1
    return count


def check_matrix(rows, metrics, methods):
    """独立两动作收益核验全部单元格，包括概率断点的左侧极限。"""
    for a in methods:
        for b in methods:
            thresholds = {0., 1., *(r['conditions'][m]['p_yes'] for r in rows for m in (a, b))}
            thresholds |= {math.nextafter(t, -math.inf) for t in thresholds if t > 0}
            gaps = []
            for t in thresholds:
                differences = []
                for row in rows:
                    y = row['label']
                    pa, pb = (row['conditions'][m]['p_yes'] for m in (a, b))
                    ua = t-y if pa <= t else y-t
                    ub = t-y if pb <= t else y-t
                    differences.append(ua - ub)
                gaps.append(sum(differences) / len(rows))
            require(abs(metrics['infogap'][a][b]['value'] - max(0., *gaps)) < 1e-12,
                    'Independent two-action payoff verification failed.')


def audit(config, run, tokenizer, old_rows, manifest):
    require(manifest['status'] == 'complete', 'Run is incomplete.')
    source = ROOT / 'runs' / config['source_run']
    old_identity = read_json(source / 'manifest.json')['identity']
    receipt = manifest['identity']['extension']
    rows = read_json(run / 'predictions.json')
    unchanged = verify_old_conditions(rows, old_rows, receipt['old_methods'])
    checked_api, checked_local = set(), set()
    for row in rows:
        require(row['conditions']['full']['text'] == row['context'], 'Full context changed.')
        require(row['conditions']['no_context']['text'] == '', 'No-context control changed.')
        for method, c in row['conditions'].items():
            folder = run if method == METHOD else source
            checked_api.add((str(folder), check_cached_prediction(folder, config, row, c)))
            require(c['tokens'] == tokenizer.count(c['text']), 'Token count differs.')
            require(c['predicted_label'] == int(c['p_yes'] >= .5), 'Predicted label differs.')
            require(c['retained_ratio_actual'] == c['tokens'] / row['original_tokens'], 'Ratio differs.')
            if method in {'full', 'no_context'}:
                continue
            budget = max(1, math.floor(config['retained_ratio'] * row['original_tokens']))
            require(c['budget_tokens'] == budget and c['budget_met'] == (c['tokens'] <= budget), 'Budget differs.')
            key = c['local_compression_hash']
            entry = read_json(folder / 'local_compression_cache' / f'{key}.json')
            identity = manifest['identity'] if method == METHOD else old_identity
            allowed = {'context': row['context'], 'budget_tokens': budget, 'method': method,
                       'retained_ratio': config['retained_ratio'], 'runtime': identity['local_compressors']}
            require(entry['request'] == allowed and fingerprint(allowed) == key, 'Compressor input isolation failed.')
            require(entry['result_sha256'] == fingerprint(entry['result']), 'Local output digest differs.')
            require(entry['result']['text'] == c['text'], 'Compressed output changed.')
            require(c['compression_request_hashes'] == [], 'Local compression unexpectedly called API.')
            checked_local.add((str(folder), key))
    metrics = read_json(run / 'metrics.json')
    check_matrix(rows, metrics, config['methods'])
    # 正式全量追加时，旧4种方法的指标及旧4×4矩阵也必须精确保持不变。
    if len(rows) == receipt['source_samples']:
        old_metrics = read_json(source / 'metrics.json')
        for method in receipt['old_methods']:
            require(metrics['scores'][method] == old_metrics['scores'][method], 'Historical scores changed.')
            for other in receipt['old_methods']:
                require(metrics['infogap'][method][other] == old_metrics['infogap'][method][other],
                        'Historical InfoGap cell changed.')
    result = {'status': 'passed', 'samples': len(rows), 'forecasts': len(rows)*len(config['methods']),
              'unchanged_forecasts_from_previous_run': unchanged, 'source_run': config['source_run'],
              'unique_api_responses_checked': len(checked_api), 'local_compressions_checked': len(checked_local),
              'infogap_cells_independently_checked': len(config['methods'])**2, 'new_api_calls': 0,
              'checks': ['source immutable files and raw dataset', 'complete request identity and raw probability',
                         'query-blind compression cache inputs', 'identical historical conditions',
                         'actual Qwen token counts', 'independent two-action payoff verification']}
    write_json(run / 'audit.json', result)
    write_json(run / 'extension_receipt.json', {**receipt, 'verification': result})
    print(result, flush=True)


def replay(config, run, tokenizer, old_rows, manifest):
    require(manifest['status'] == 'complete', 'Run is incomplete.')
    names = ('predictions.json', 'metrics.json', 'api_attempts.json')
    before = {n: file_hash(run / n) for n in names}
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')), \
         patch('blackwell.local_cache.create_compressor', side_effect=AssertionError('Model loading forbidden')):
        rows, execution = predict_rows(config, run, tokenizer, old_rows, manifest, offline=True)
        write_outputs(run, rows, config)
    after = {n: file_hash(run / n) for n in names}
    require(before == after and execution['new_api_attempts'] == 0, 'Offline replay changed results or ledger.')
    result = {'status': 'passed', 'new_api_attempts': 0, 'socket_connections_blocked': True,
              'local_model_loading_blocked': True, 'cache_hits': execution['cache_hits'],
              'byte_identical_files_sha256': after}
    write_json(run / 'replay_verification.json', result)
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=['prepare', 'compress', 'predict', 'audit', 'replay'], default='predict')
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    # 准备阶段只能读取本地固定数据/权重，不能意外下载或调用 API。
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Preparation must be offline')):
        run, tokenizer, old_rows, manifest = prepare(config)
    if args.stage == 'prepare':
        print(f'Prepared {len(old_rows)} frozen samples: {run}', flush=True)
    elif args.stage == 'compress':
        compact_only(config, run, tokenizer, old_rows, manifest)
    elif args.stage == 'audit':
        audit(config, run, tokenizer, old_rows, manifest)
    elif args.stage == 'replay':
        replay(config, run, tokenizer, old_rows, manifest)
    else:
        load_local_keys()
        manifest['status'] = 'running'
        write_json(run / 'manifest.json', manifest)
        try:
            rows, execution = predict_rows(config, run, tokenizer, old_rows, manifest)
            write_outputs(run, rows, config)
            manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(), last_execution=execution)
        except Exception as error:
            manifest.update(status='failed', error_type=type(error).__name__)
            raise
        finally:
            write_json(run / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
