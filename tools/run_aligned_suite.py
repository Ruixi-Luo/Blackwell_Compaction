"""预算对齐的新实验：固定旧样本 → 本地压缩 → API 改写 → 全部重新预测 → 审计。

各阶段可续跑；旧目录只读。示例：
python tools/run_aligned_suite.py --config configs/aligned_smoke.json --stage all
"""
import argparse
import gc
import math
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blackwell.budgeting import align_local, select_candidate
from blackwell.client import CachedChatClient, parse_probability
from blackwell.compressors import LOCAL_METHODS, create_compressor
from blackwell.compressors.llm_rewrite import API_METHODS, LLMRewriteCompressor, rewrite_messages
from blackwell.diagnostics import probability_diagnostics, retention_diagnostics
from blackwell.prompts import prediction_messages
from blackwell.storage import file_hash, fingerprint, read_json, write_json
from run_experiment import code_provenance, load_config, load_local_keys, write_outputs

METHODS = ['full', 'no_context', 'head_tail', *LOCAL_METHODS, *API_METHODS]


def compression_config(config):
    """只覆盖生成器的模型、超时和尝试上限；不修改 reader 配置。"""
    return {**config, 'model': config['compression_model'],
            'max_api_attempts': config['compression_max_api_attempts'],
            'timeout_seconds': config.get('compression_timeout_seconds', 600)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(config):
    """固定既有样本和源文件哈希；重新预测全部条件，避免混用旧日期的 forecasts。"""
    from blackwell.data import load_dataset
    from blackwell.compressors.runtime import configure_cache, runtime_identity, verify_assets
    from blackwell.tokenization import ContextTokenizer
    configure_cache()
    verify_assets(LOCAL_METHODS)
    require(config['methods'] == METHODS, 'Unexpected paired methods.')
    require(config['retained_ratio'] == .5, 'This protocol fixes 50% token cap.')
    source = ROOT / 'runs' / config['source_run']
    require(source.parent == ROOT / 'runs' and source.name != config['run_name'], 'Unsafe source path.')
    old = read_json(source / 'manifest.json')
    require(old['status'] == 'complete', 'Source run incomplete.')
    examples = read_json(source / 'examples.json')
    reloaded, provenance = load_dataset(old['identity']['config'], offline=True)
    require(examples == reloaded and fingerprint(examples) == old['identity']['examples_sha256'], 'Source samples changed.')
    require(config['sample_size'] <= len(examples), 'Cannot expand the frozen source population.')
    examples = examples[:config['sample_size']]
    snapshots = {p.name: file_hash(p) for p in source.iterdir() if p.is_file()}
    code = code_provenance()
    for path in [Path(__file__), *sorted((ROOT / 'blackwell/compressors/prompts').glob('*'))]:
        code['files_sha256'][path.relative_to(ROOT).as_posix()] = file_hash(path)
    runtime = runtime_identity(LOCAL_METHODS, config['compression_device'])
    runtime['budget_policy'] = 'length-only native-rate search <=8 calls; prefer 95-100% cap band, else closest length (feasible wins ties); explicitly head-tail clamp selected overshoots'
    identity = {'config': config, 'dataset': provenance, 'examples_sha256': fingerprint(examples),
                'code': code['files_sha256'], 'source_files_sha256': snapshots,
                'local_compressors': runtime}
    run = ROOT / 'runs' / config['run_name']
    if (run / 'manifest.json').exists():
        manifest = read_json(run / 'manifest.json')
        require(manifest['identity'] == identity, 'Code/config/data changed: choose a new run_name.')
        require(read_json(run / 'examples.json') == examples, 'Saved examples changed.')
    else:
        manifest = {'identity': identity, 'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'prepared'}
        write_json(run / 'manifest.json', manifest)
        write_json(run / 'examples.json', examples)
    return run, ContextTokenizer(), examples, manifest


def compression_request(context, budget, method, manifest):
    """压缩缓存键只含上下文、预算和固定协议；绝不传整个 example。"""
    identity = manifest['identity']
    return {'context': context, 'budget': budget, 'method': method,
            'protocol_sha256': fingerprint({'code': identity['code'],
                                           'runtime': identity['local_compressors'],
                                           'model': identity['config']['compression_model']})}


def read_compaction(run, request):
    key = fingerprint(request)
    path = run / 'aligned_compression_cache' / f'{key}.json'
    if not path.exists():
        return None
    entry = read_json(path)
    require(entry['request'] == request and entry['sha256'] == fingerprint(entry['result']), 'Compression cache corruption.')
    return {**entry['result'], 'local_compression_hash': key}


def save_compaction(run, request, result, seconds):
    result = {**result, 'compression_seconds': seconds}
    write_json(run / 'aligned_compression_cache' / f'{fingerprint(request)}.json',
               {'request': request, 'result': result, 'sha256': fingerprint(result)})


def compress_stage(config, run, tokenizer, examples, manifest, api=False):
    """先按方法遍历，GPU 一次只装一个模型；相同原文只压缩一次。"""
    methods = API_METHODS if api else LOCAL_METHODS
    contexts = list(dict.fromkeys(e['context'] for e in examples))
    api_config = compression_config(config)
    client = CachedChatClient(api_config, run / 'compression_api') if api else None
    for method in methods:
        engine = None
        for index, context in enumerate(contexts, 1):
            budget = max(1, math.floor(tokenizer.count(context)*config['retained_ratio']))
            request = compression_request(context, budget, method, manifest)
            if read_compaction(run, request) is not None:
                continue
            if engine is None:
                # LLMLingua 的 seed_everything 会改环境字符串，但不会改变已启动进程的 hash seed。
                # 先核实进程从 seed=0 启动，再恢复字符串，供 SALT 的防误用检查读取。
                require(not sys.flags.hash_randomization, 'Start with PYTHONHASHSEED=0.')
                os.environ['PYTHONHASHSEED'] = '0'
                engine = LLMRewriteCompressor(client, tokenizer, method) if api else create_compressor(method, config['compression_device'])
            start = time.perf_counter()
            result = engine.compress(context, budget) if api else align_local(engine, context, budget, tokenizer)
            save_compaction(run, request, result, time.perf_counter()-start)
            print(f'{method} [{index}/{len(contexts)}] {result["tokens"]}/{budget} tokens; clip={result["post_clip_used"]}', flush=True)
        del engine
        gc.collect()
        if not api:
            import torch
            torch.cuda.empty_cache()


def condition_text(context, method, tokenizer, run, manifest):
    """预测阶段只读已完成的压缩结果，不再加载压缩模型或修改压缩文本。"""
    original = tokenizer.count(context)
    budget = max(1, math.floor(.5*original))
    if method in LOCAL_METHODS or method in API_METHODS:
        result = read_compaction(run, compression_request(context, budget, method, manifest))
        require(result is not None, f'Missing compression: {method}')
        # 候选全文保存在压缩缓存，不在每道问题中重复复制。
        return {k: v for k, v in result.items() if k != 'candidates'}
    text = context if method == 'full' else ''
    if method == 'head_tail':
        text = tokenizer.head_tail(context, budget)
        return select_candidate([{'text': text, 'tokens': tokenizer.count(text)}], budget, tokenizer)
    return {'text': text, 'tokens': tokenizer.count(text), 'budget_tokens': None,
            'budget_met': None, 'length_repair_used': False, 'compression_request_hashes': []}


def predict(config, run, tokenizer, examples, manifest, offline=False):
    client = CachedChatClient(config, run, offline=offline)
    rows = []
    # 全部压缩完成后才访问 question / label；每题的所有条件配对。
    for index, example in enumerate(examples, 1):
        context = example['context']
        original = tokenizer.count(context)
        conditions = {}
        for method in config['methods']:
            c = condition_text(context, method, tokenizer, run, manifest)
            raw, key = client.complete(prediction_messages(c['text'], example['question']), config['prediction_max_tokens'], json_mode=True)
            probability = parse_probability(raw)
            conditions[method] = {**c, 'p_yes': probability, 'predicted_label': int(probability >= .5),
                                  'prediction_request_hash': key, 'retained_ratio_actual': c['tokens']/original}
        rows.append({**example, 'original_tokens': original, 'conditions': conditions})
        if not offline:
            write_json(run / 'predictions.json', rows)
            print(f'Predict [{index}/{len(examples)}] '+example['id'], flush=True)
    return rows


def finish(config, run, rows, manifest):
    metrics = write_outputs(run, rows, config)
    labels = [row['label'] for row in rows]
    diagnostics = {}
    for method in config['methods']:
        conditions = [row['conditions'][method] for row in rows]
        diagnostics[method] = {'probability': probability_diagnostics([c['p_yes'] for c in conditions], labels),
                               'retention': retention_diagnostics(conditions)}
    write_json(run / 'diagnostics.json', diagnostics)
    manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(run / 'manifest.json', manifest)
    return metrics


def audit(config, run, tokenizer, examples, manifest):
    """重新构造每条 API 请求，核验长度与 query 边界，并独立验证全部 InfoGap 单元格。"""
    from tools.extend_with_salt import check_cached_prediction, check_matrix
    rows = read_json(run / 'predictions.json')
    require(len(rows) == len(examples), 'Incomplete sample set.')
    source = ROOT / 'runs' / config['source_run']
    for name, digest in manifest['identity']['source_files_sha256'].items():
        require(file_hash(source / name) == digest, 'Old source file changed: '+name)
    checked = set()
    for row, example in zip(rows, examples):
        require({k: row[k] for k in example} == example, 'Sample pairing changed.')
        require(list(row['conditions']) == config['methods'], 'Condition mismatch.')
        for method, condition in row['conditions'].items():
            require(tokenizer.count(condition['text']) == condition['tokens'], 'Token count mismatch.')
            check_cached_prediction(run, config, row, condition)
            expected = condition_text(row['context'], method, tokenizer, run, manifest)
            require(all(condition[k] == v for k, v in expected.items()), 'Condition differs from compression cache.')
            if method in {'full', 'no_context', 'head_tail'}:
                continue
            request = compression_request(row['context'], condition['budget_tokens'], method, manifest)
            cached = read_compaction(run, request)
            key = fingerprint(request)
            if key in checked:
                continue
            checked.add(key)
            candidates = cached['candidates']
            require(all(tokenizer.count(c['text']) == c['tokens'] for c in candidates), 'Candidate length mismatch.')
            selection = select_candidate(candidates, condition['budget_tokens'], tokenizer)
            require(all(cached[k] == v for k, v in selection.items() if k != 'compression_request_hashes'), 'Length-only candidate selection mismatch.')
            if method in API_METHODS:
                previous, previous_tokens = None, None
                for candidate in candidates:
                    messages = rewrite_messages(row['context'], condition['budget_tokens'], method, previous, previous_tokens)
                    raw, request_hash = CachedChatClient(compression_config(config), run / 'compression_api', offline=True).complete(
                        messages, max_tokens=min(32768, max(512, 2*condition['budget_tokens']+512)))
                    require(raw == candidate['text'] and request_hash == candidate['request_hash'], 'Rewrite request/response mismatch.')
                    previous, previous_tokens = raw, candidate['tokens']
    check_matrix(rows, read_json(run / 'metrics.json'), config['methods'])
    before = {name: file_hash(run / name) for name in ('predictions.json', 'metrics.json', 'api_attempts.json')}
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline replay')):
        replayed = predict(config, run, tokenizer, examples, manifest, offline=True)
    require(replayed == rows, 'Offline predictions changed.')
    require(before == {name: file_hash(run / name) for name in before}, 'Replay changed files.')
    result = {'status': 'passed', 'samples': len(rows), 'forecasts': len(rows)*len(METHODS),
              'compression_contexts_checked': len(checked), 'infogap_cells': len(METHODS)**2,
              'old_files_unchanged': True, 'offline_replay_identical': True,
              'scope': 'Full request reconstruction; compression inputs contain context/length only; original forecasts unmodified.'}
    write_json(run / 'audit.json', result)
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=['prepare', 'local', 'rewrite', 'predict', 'audit', 'all'], default='all')
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Preparation must be offline')):
        run, tokenizer, examples, manifest = prepare(config)
    load_local_keys()
    try:
        if args.stage in {'all', 'local'}:
            manifest['status'] = 'local_compression'
            write_json(run / 'manifest.json', manifest)
            with patch.object(socket.socket, 'connect', side_effect=AssertionError('Local compression must be offline')):
                compress_stage(config, run, tokenizer, examples, manifest)
        if args.stage in {'all', 'rewrite'}:
            manifest['status'] = 'api_compression'
            write_json(run / 'manifest.json', manifest)
            compress_stage(config, run, tokenizer, examples, manifest, api=True)
        if args.stage in {'all', 'predict'}:
            manifest['status'] = 'predicting'
            write_json(run / 'manifest.json', manifest)
            rows = predict(config, run, tokenizer, examples, manifest)
            finish(config, run, rows, manifest)
        if args.stage in {'all', 'audit'}:
            audit(config, run, tokenizer, examples, manifest)
    except Exception as error:
        manifest.update(status='failed', error_type=type(error).__name__)
        write_json(run / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()
