"""主流程：准备数据 -> 压缩 -> 预测 -> 评估；每步都能单独执行。"""
import gc
import math
import sys
import time
from datetime import datetime, timezone
from blackwell.storage import ROOT, read_json, write_json, fingerprint, file_hash
from blackwell.tokenization import ContextTokenizer, TOKENIZER_SHA256
from blackwell.client import CachedChatClient, parse_probability
from blackwell.prompts import prediction_messages
from blackwell.diagnostics import probability_diagnostics
from .config import code_hashes, package_versions
from .data import load_examples
from .api import make_client, predict_one, request_identity
from .compression import (LOCAL_METHODS, load_compression, save_compression,
                          create_engine, compress_one, compression_path)
from .summary import summary_messages, output_ceiling
from .evaluation import analyze


def prepare(config, run, offline=False):
    """固定样本、配置、代码身份；offline=True 时连数据/tokenizer 也不下载。"""
    tokenizer_path = ROOT/'data/tokenizer/qwen3-30b-a3b-instruct-2507.json'
    if offline and not tokenizer_path.exists():
        raise RuntimeError('Offline tokenizer missing; run prepare without --offline first.')
    tokenizer = ContextTokenizer()
    identity = {'config': config, 'code': code_hashes(), 'tokenizer_sha256': TOKENIZER_SHA256, 'packages': package_versions()}
    path = run/'manifest.json'
    if path.exists():
        manifest = read_json(path)
        if manifest['identity'] != identity:
            raise ValueError('Frozen code/config changed. Choose a new run_name; do not overwrite old results.')
        for name in config['datasets']:
            if fingerprint(read_json(run/name/'examples.json')) != manifest['datasets'][name]['examples_sha256']:
                raise ValueError('Saved examples changed.')
        return tokenizer, manifest
    manifest = {'identity': identity, 'status': 'prepared', 'datasets': {},
                'created_utc': datetime.now(timezone.utc).isoformat()}
    # 数据准备完成才写 manifest；失败时不伪造完整运行。
    for name in config['datasets']:
        examples, source = load_examples(name, config['sample_size'], config['sample_seed'], offline)
        write_json(run/name/'examples.json', examples)
        manifest['datasets'][name] = {'source': source, 'n': len(examples),
            'yes': sum(e['label'] for e in examples), 'examples_sha256': fingerprint(examples)}
    write_json(path, manifest)
    return tokenizer, manifest


def budget_for(context, tokenizer, config):
    """原文 Qwen token 数乘固定保留率，向下取整；与其他方法输出无关。"""
    budget = math.floor(tokenizer.count(context)*config['retained_ratio'])
    if budget < 1:
        raise ValueError('Source is too short for a positive budget.')
    return budget


def compress(config, run, tokenizer, offline=False, allow_api=False):
    """逐方法运行，一个 GPU 模型用完即释放；相同正文只压缩一次。"""
    for method in config['methods']:
        if method in ('full', 'no_context'):
            continue
        engine = client = None
        for name in config['datasets']:
            folder = run/name
            contexts = dict.fromkeys(e['context'] for e in read_json(folder/'examples.json'))
            for i, context in enumerate(contexts, 1):
                budget = budget_for(context, tokenizer, config)
                if load_compression(folder, context, method, budget) is not None:
                    continue
                if offline:
                    raise RuntimeError('Offline compression cache missing; no GPU/API call made.')
                if method in LOCAL_METHODS and engine is None:
                    if sys.flags.hash_randomization:
                        raise RuntimeError('Start CMD with: set PYTHONHASHSEED=0')
                    from blackwell.compressors.runtime import verify_assets
                    if method != 'dac':
                        verify_assets([method])
                    engine = create_engine(method, config['device'])
                if method == 'llm_summary' and client is None:
                    client = make_client(config, 'summary', run, tokenizer, allow_api)
                if engine is not None and config['device'] == 'cuda':
                    import torch
                    torch.cuda.synchronize()
                start = time.perf_counter()
                result = compress_one(context, budget, method, tokenizer, engine, client, config['retained_ratio'])
                if engine is not None and config['device'] == 'cuda':
                    torch.cuda.synchronize()
                result['compression_seconds'] = time.perf_counter()-start
                save_compression(folder, context, method, budget, result)
                print(f'compress {name} {method} {i}/{len(contexts)}: {result["tokens"]}/{budget}', flush=True)
        del engine, client
        gc.collect()
        if 'torch' in sys.modules and config['device'] == 'cuda':
            sys.modules['torch'].cuda.empty_cache()


def condition_for(folder, context, method, tokenizer, config):
    """将 Full、空上下文和缓存压缩文本统一成后续 reader 使用的格式。"""
    if method in ('full', 'no_context'):
        text = context if method == 'full' else ''
        return {'text': text, 'tokens': tokenizer.count(text), 'budget_tokens': None,
                'budget_met': None, 'length_repair_used': False}
    result = load_compression(folder, context, method, budget_for(context, tokenizer, config))
    if result is None:
        raise RuntimeError('Run the compress stage before predict.')
    return {**result, 'local_compression_hash': fingerprint({'context': context,
            'budget_tokens': budget_for(context, tokenizer, config), 'method': method})}


def predict(config, run, tokenizer, allow_api=False):
    """同一题目上的所有条件都预测；缓存先写盘，意外中断后可续跑。"""
    client = make_client(config, 'reader', run, tokenizer, allow_api)
    for name in config['datasets']:
        folder = run/name
        examples = read_json(folder/'examples.json')
        rows = []
        for i, example in enumerate(examples, 1):
            conditions = {}
            original_tokens = tokenizer.count(example['context'])
            for method in config['methods']:
                condition = condition_for(folder, example['context'], method, tokenizer, config)
                forecast = predict_one(condition['text'], example['question'], client, config['reader']['max_tokens'])
                conditions[method] = {**condition, **forecast,
                    'retained_ratio_actual': condition['tokens']/original_tokens}
            rows.append({**example, 'original_tokens': original_tokens, 'conditions': conditions})
            write_json(folder/'predictions.json', rows)
            if i % 20 == 0 or i == len(examples):
                print(f'predict {name} {i}/{len(examples)}', flush=True)


def evaluate(config, run):
    """仅读取保存的概率；不会请求模型，也不会改变原始 p_yes。"""
    for name in config['datasets']:
        folder = run/name
        rows = read_json(folder/'predictions.json')
        metrics = analyze(rows, config)
        write_json(folder/'metrics.json', metrics)
        write_json(folder/'diagnostics.json', {method: probability_diagnostics(
            [r['conditions'][method]['p_yes'] for r in rows], [r['label'] for r in rows])
            for method in config['methods']})


def check_api_record(run, role, key, expected):
    """重建请求后核验缓存键及完整响应，返回真实模型文本。"""
    path = run/'api'/role/'api_cache'/f'{key}.json'
    record = read_json(path)
    if fingerprint(expected) != key or record['request'] != expected:
        raise ValueError('Saved API request does not match source/prompt/settings.')
    return CachedChatClient._extract(record['response'], key)[0]


def audit(config, run, tokenizer, manifest):
    """独立核验全部题目、真实请求、预算、指标和导入来源；失败不生成通过标记。"""
    for name in config['datasets']:
        folder = run/name
        rows, examples = read_json(folder/'predictions.json'), read_json(folder/'examples.json')
        if len(rows) != len(examples):
            raise ValueError('Incomplete predictions.')
        for row, example in zip(rows, examples):
            if any(row[k] != v for k, v in example.items()):
                raise ValueError('Prediction row no longer matches the sampled question.')
            for method in config['methods']:
                c = row['conditions'][method]
                expected_condition = condition_for(folder, row['context'], method, tokenizer, config)
                if any(c[k] != v for k, v in expected_condition.items()):
                    raise ValueError('Prediction uses a different compression output.')
                if tokenizer.count(c['text']) != c['tokens']:
                    raise ValueError('Incorrect token count.')
                identity = request_identity(config['reader'], prediction_messages(c['text'], row['question']),
                                            config['reader']['max_tokens'], True)
                raw = check_api_record(run, 'reader', c['prediction_request_hash'], identity)
                if parse_probability(raw) != c['p_yes'] or c['predicted_label'] != int(c['p_yes'] >= .5):
                    raise ValueError('Probability differs from its raw API response.')
                if method in ('full', 'no_context'):
                    continue
                budget = budget_for(row['context'], tokenizer, config)
                if c['budget_met'] != (c['tokens'] <= budget) or c['post_clip_used'] or c['matching_trim_used']:
                    raise ValueError('Budget metadata or no-clipping rule violated.')
                if method in LOCAL_METHODS and c['tokens'] > budget:
                    raise ValueError('Local method exceeded the hard cap.')
                if method == 'llm_summary':
                    key = c.get('request_hash', c.get('original_request_hash'))
                    record = read_json(run/'api/summary/api_cache'/f'{key}.json')
                    # 导入的历史非流式请求保留其真实身份，不把它伪装成流式。
                    api = {**config['summary'], 'stream': record['request']['payload']['stream']}
                    expected = request_identity(api, summary_messages(row['context'], budget), output_ceiling(budget))
                    if check_api_record(run, 'summary', key, expected) != c['text']:
                        raise ValueError('Summary is not the original-prompt complete response.')
        if analyze(rows, config) != read_json(folder/'metrics.json'):
            raise ValueError('Recomputed metrics differ.')
        write_json(folder/'audit.json', {'status': 'passed', 'questions': len(rows),
            'forecasts': len(rows)*len(config['methods']), 'matrix_cells': len(config['methods'])**2,
            'query_blind_compression': True, 'original_first_draft': True,
            'predictions_sha256': file_hash(folder/'predictions.json'),
            'metrics_sha256': file_hash(folder/'metrics.json'),
            'summary_over_target': sum(r['conditions'].get('llm_summary', {}).get('budget_met') is False for r in rows)})
    provenance_path = run/'import_provenance.json'
    if provenance_path.exists():
        for relative, digest in read_json(provenance_path)['source_files'].items():
            if file_hash(ROOT/relative) != digest:
                raise ValueError('Historical source changed: '+relative)
    manifest['status'] = 'complete'
    write_json(run/'manifest.json', manifest)
    print('Audit passed: real responses, paired examples, query-blind requests and both InfoGap directions.', flush=True)
