"""一次性旧结果适配器：仅用于核验，不是从头运行的前置依赖。"""
import shutil
from pathlib import Path
from blackwell.storage import ROOT, read_json, write_json, fingerprint, file_hash
from blackwell.client import CachedChatClient, parse_probability
from blackwell.prompts import prediction_messages
from .api import request_identity
from .summary import summary_messages, output_ceiling
from .pipeline import budget_for
from .compression import compression_path, save_compression


def import_firstdraft(source, run, config, tokenizer):
    """source：已审计旧目录；run：新的独立目录；只复制核验通过的完整文本/响应。"""
    source = source.resolve()
    if not source.is_relative_to((ROOT/'runs').resolve()) or source == run.resolve():
        raise ValueError('Import source must be a distinct local run directory.')
    marker = run/'import_provenance.json'
    if marker.exists():
        if read_json(marker)['source'] != source.relative_to(ROOT).as_posix():
            raise ValueError('Already imported a different source.')
        return
    manifest = read_json(source/'manifest.json')
    if manifest['status'] != 'complete':
        raise ValueError('Only completed audited sources may be imported.')
    old = manifest['config']
    for key in ('sample_size', 'sample_seed', 'retained_ratio', 'methods', 'bootstrap_seed', 'bootstrap_repetitions'):
        if old[key] != config[key]:
            raise ValueError('Migration protocol mismatch: '+key)
    source_files = {}
    attempt_sources = {"reader": set(), "summary": set()}
    def record(path):
        source_files[path.relative_to(ROOT).as_posix()] = file_hash(path)
    def copy_api(path, role, expected):
        entry = read_json(path)
        if entry['request'] != expected or path.stem != fingerprint(expected):
            raise ValueError('Historical request identity mismatch.')
        CachedChatClient._extract(entry['response'], path.stem)
        target = run/'api'/role/'api_cache'/path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and file_hash(target) != file_hash(path):
            raise ValueError('Refusing to overwrite a different cached response.')
        shutil.copy2(path, target)
        record(path)
        attempt_sources[role].add(path.parent.parent/'api_attempts.json')
    record(source/'manifest.json')
    for name in config['datasets']:
        folder = source/name
        audit = read_json(folder/'audit.json')
        if audit['status'] != 'passed' or not audit['original_prompt_first_draft']:
            raise ValueError('Source is not an audited first-draft study.')
        examples = read_json(run/name/'examples.json')
        if examples != read_json(folder/'examples.json'):
            raise ValueError('Freshly sampled examples differ from historical examples.')
        for filename in ('audit.json', 'examples.json', 'metrics.json', 'predictions.json'):
            record(folder/filename)
        for context in dict.fromkeys(e['context'] for e in examples):
            budget = budget_for(context, tokenizer, config)
            for method in config['methods']:
                if method in ('full', 'no_context'):
                    continue
                path = compression_path(folder, context, method)
                entry = read_json(path)
                if entry['request'] != {'context': context, 'budget_tokens': budget, 'method': method}:
                    raise ValueError('Historical compression input mismatch.')
                if fingerprint(entry['result']) != entry['result_sha256']:
                    raise ValueError('Historical compression result hash mismatch.')
                result = entry['result']
                if result['tokens'] != tokenizer.count(result['text']) or result['post_clip_used'] or result['matching_trim_used']:
                    raise ValueError('Invalid or clipped historical compression.')
                if method == 'llm_summary':
                    candidate = entry['candidates'][0]
                    if not candidate['complete_verified'] or candidate.get('revision') or candidate.get('output_limit_retry'):
                        raise ValueError('Selected summary is not a verified original first draft.')
                    if candidate['text'] != result['text'] or result['selected_candidate'] != 0:
                        raise ValueError('Summary output differs from the first candidate.')
                    api_path = ROOT/candidate['api_folder']/'api_cache'/f"{candidate['request_hash']}.json"
                    actual = read_json(api_path)['request']
                    api = {**config['summary'], 'stream': actual['payload']['stream']}
                    expected = request_identity(api, summary_messages(context, budget), output_ceiling(budget))
                    copy_api(api_path, 'summary', expected)
                    if CachedChatClient._extract(read_json(api_path)['response'], api_path.stem)[0] != result['text']:
                        raise ValueError('Original summary differs from API response.')
                elif result['tokens'] > budget:
                    raise ValueError('Historical local output exceeds cap.')
                record(path)
                save_compression(run/name, context, method, budget, result,
                    {'source': path.relative_to(ROOT).as_posix(), 'sha256': file_hash(path),
                     'note': 'Imported historical output; not a newly computed compression.'})
        for row in read_json(folder/'predictions.json'):
            for method in config['methods']:
                c = row['conditions'][method]
                key = c['prediction_request_hash']
                expected = request_identity(config['reader'], prediction_messages(c['text'], row['question']), 128, True)
                copy_api(folder/'api_cache'/f'{key}.json', 'reader', expected)
                if parse_probability(CachedChatClient._extract(read_json(folder/'api_cache'/f'{key}.json')['response'], key)[0]) != c['p_yes']:
                    raise ValueError('Historical probability differs from raw response.')
    # 费用和尝试次数都继承，防止换目录重置上限；相同记录只计一次。
    for role, ledgers in attempt_sources.items():
        attempts = {}
        for ledger in sorted(ledgers):
            if ledger.exists():
                record(ledger)
                for item in read_json(ledger):
                    attempts[fingerprint(item)] = item
        write_json(run/'api'/role/'api_attempts.json', list(attempts.values()))
    spending = source/'spending_reservations.json'
    if (run/'spending_reservations.json').exists():
        raise ValueError('Import into a fresh run; do not reset an existing cost ledger.')
    shutil.copy2(spending, run/'spending_reservations.json')
    record(spending)
    write_json(marker, {'source': source.relative_to(ROOT).as_posix(), 'source_files': source_files,
        'new_api_calls': 0, 'new_gpu_compressions': 0,
        'historical_cost_inherited_cny': sum(x['accounted_cny'] for x in read_json(spending)),
        'scope': 'Exact source texts and API identities reused; recompute forecasts and metrics offline.'})
    print('Imported verified first drafts and exact reader responses; no model calls.', flush=True)
