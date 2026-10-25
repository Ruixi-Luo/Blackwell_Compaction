"""Reuse the completed cap study; replace only summary with its original first draft.

The summary's 50% length is a requested target, not an enforced output cap.
No compression requests, editing, truncation, or length-based selection occur.
"""
import argparse
import copy
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blackwell.storage import read_json, write_json, fingerprint, file_hash
from blackwell.tokenization import ContextTokenizer
from blackwell.client import CachedChatClient, parse_probability
from blackwell.prompts import prediction_messages
from blackwell.cost_guard import CostGuard, GuardedClient
from blackwell.diagnostics import probability_diagnostics
from run_experiment import load_local_keys, analyze
from tools.run_mini100_cap import entry_path, payload, verify_summary_candidate

METHODS = ['full', 'no_context', 'selective_context', 'llmlingua2', 'llm_summary', 'dac']
DATASETS = ['hotpotqa', 'qasper']
SOURCE = ROOT/'runs/mini100_cap50_20261025_v4'
SUITE = ROOT/'runs/mini100_firstdraft_20261025_v5'
CONFIG = ROOT/'configs/mini100_firstdraft.json'


def select_original_first(candidates):
    """Select position zero; a longer-than-target first draft stays selected."""
    if not candidates:
        raise ValueError('Missing original draft.')
    first = candidates[0]
    if not first.get('complete_verified') or 'origin_file' not in first or first.get('revision'):
        raise ValueError('First draft must be verified, complete and from the original prompt.')
    if first.get('output_limit_retry'):
        raise ValueError('A regenerated completion is not a one-call first draft.')
    return first


def core_hashes():
    paths = [Path(__file__), CONFIG, ROOT/'tools/run_mini100_cap.py', ROOT/'run_experiment.py',
             *sorted((ROOT/'blackwell').rglob('*.py'))]
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}


def prepare(config):
    tokenizer = ContextTokenizer()
    if (SUITE/'manifest.json').exists():
        manifest = read_json(SUITE/'manifest.json')
        if manifest['config'] != config or manifest['core_hashes'] != core_hashes():
            raise ValueError('Frozen code/configuration changed.')
        return tokenizer, manifest
    old_manifest = read_json(SOURCE/'manifest.json')
    if old_manifest['status'] != 'complete' or any(read_json(SOURCE/n/'audit.json')['status'] != 'passed' for n in DATASETS):
        raise ValueError('Wait for the cap study to finish its audit; do not duplicate its active reader.')
    old_files = dict(old_manifest['old_files_sha256'])
    for path in SOURCE.rglob('*'):
        if path.is_file():
            old_files[path.relative_to(ROOT).as_posix()] = file_hash(path)
    for name in DATASETS:
        folder = SUITE/name
        write_json(folder/'examples.json', read_json(SOURCE/name/'examples.json'))
        write_json(folder/'config.json', {**config, 'dataset': name})
        receipts = {}
        for path in (SOURCE/name/'api_cache').glob('*.json'):
            dest = folder/'api_cache'/path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            receipts[path.stem] = {'source': path.relative_to(ROOT).as_posix(), 'sha256': file_hash(path)}
        write_json(folder/'reader_reuse.json', receipts)
        # Keep cumulative reader attempt counts as well as the shared currency ledger.
        shutil.copy2(SOURCE/name/'api_attempts.json', folder/'api_attempts.json')
        for path in (SOURCE/name/'compressions').glob('*.json'):
            entry = read_json(path)
            if entry['request']['method'] == 'llm_summary':
                first = select_original_first(entry['candidates'])
                budget = entry['request']['budget_tokens']
                verify_summary_candidate(first, [], entry['request']['context'], budget, config, tokenizer)
                result = {**entry['result'], 'text': first['text'], 'tokens': first['tokens'],
                          'budget_met': first['tokens'] <= budget, 'budget_fill': first['tokens']/budget,
                          'within_target_band': .95*budget <= first['tokens'] <= budget,
                          'length_repair_used': False, 'selected_candidate': 0,
                          'selection_policy': 'original_prompt_first_draft',
                          'compression_request_hashes': [first['request_hash']],
                          'original_request_hash': first['request_hash'],
                          'hard_cap_enforced': False, 'new_compression_calls': 0}
                entry = {'request': entry['request'], 'candidates': [first], 'result': result,
                         'result_sha256': fingerprint(result), 'source_entry_sha256': file_hash(path),
                         'source_entry': path.relative_to(ROOT).as_posix()}
            write_json(folder/'compressions'/path.name, entry)
    spending = read_json(SOURCE/'spending_reservations.json')
    if any(x['status'] == 'reserved' for x in spending):
        raise ValueError('Source has an active spending reservation.')
    write_json(SUITE/'spending_reservations.json', spending)
    shutil.copy2(SOURCE/'dac_preflight.json', SUITE/'dac_preflight.json')
    manifest = {'config': config, 'datasets': old_manifest['datasets'], 'core_hashes': core_hashes(),
                'old_files_sha256': old_files, 'status': 'prepared',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'source_suite': SOURCE.relative_to(ROOT).as_posix(),
                'scope': 'Original first-draft summary; report budget violations. Other five conditions unchanged.',
                'cost_inherited_once_cny': sum(x['accounted_cny'] for x in spending)}
    write_json(SUITE/'manifest.json', manifest)
    print('Prepared first-draft extension without any compressor API call.', flush=True)
    return tokenizer, manifest


def predict(config, tokenizer, offline=False):
    index = {p.stem: p for p in sorted((ROOT/'runs').rglob('api_cache/*.json')) if SUITE not in p.parents} if not offline else {}
    guard = None if offline else CostGuard(SUITE/'spending_reservations.json', tokenizer, config['cost_reserve_cny'])
    for name in DATASETS:
        folder = SUITE/name
        cfg = {**config, 'stream': False, 'timeout_seconds': 90, 'max_api_attempts': 800, 'min_interval_seconds': 2.}
        client = CachedChatClient(cfg, folder, offline=True) if offline else GuardedClient(cfg, folder, guard)
        receipts = read_json(folder/'reader_reuse.json')
        rows = []
        for number, original in enumerate(read_json(SOURCE/name/'predictions.json'), 1):
            row = copy.deepcopy(original)
            entry = read_json(entry_path(folder, row['context'], 'llm_summary'))
            condition = {**entry['result'], 'local_compression_hash': fingerprint(entry['request'])}
            messages = prediction_messages(condition['text'], row['question'])
            request = payload(cfg, messages, 128, True)
            key = fingerprint(request)
            dest = folder/'api_cache'/f'{key}.json'
            if not offline and not dest.exists() and key in index:
                record = read_json(index[key])
                if record['request'] != request:
                    raise ValueError('External reader cache identity mismatch.')
                parse_probability(CachedChatClient._extract(record['response'], key)[0])
                shutil.copy2(index[key], dest)
                receipts[key] = {'source': index[key].relative_to(ROOT).as_posix(), 'sha256': file_hash(index[key])}
            raw, key = client.complete(messages, 128, True)
            probability = parse_probability(raw)
            condition.update(p_yes=probability, predicted_label=int(probability >= .5), prediction_request_hash=key,
                             retained_ratio_actual=condition['tokens']/row['original_tokens'], reader_reused=key in receipts)
            row['conditions']['llm_summary'] = condition
            rows.append(row)
            if not offline:
                write_json(folder/'reader_reuse.json', receipts)
                write_json(folder/'predictions.json', rows)
                print(f'First draft {name} {number}/100', flush=True)
        if offline:
            if rows != read_json(folder/'predictions.json'):
                raise ValueError('Offline replay changed predictions.')
        else:
            write_json(folder/'metrics.json', analyze(rows, config))
            write_json(folder/'diagnostics.json', {m: probability_diagnostics([r['conditions'][m]['p_yes'] for r in rows],
                        [r['label'] for r in rows]) for m in METHODS})


def audit(config, tokenizer, manifest):
    from tools.extend_with_salt import check_cached_prediction, check_matrix
    for path, digest in manifest['old_files_sha256'].items():
        if file_hash(ROOT/path) != digest:
            raise ValueError('Historical file changed: '+path)
    for name in DATASETS:
        folder = SUITE/name
        rows, originals = read_json(folder/'predictions.json'), read_json(SOURCE/name/'predictions.json')
        if len(rows) != 100:
            raise ValueError('Incomplete study.')
        over = 0
        for row, old in zip(rows, originals):
            if {k: v for k, v in row.items() if k != 'conditions'} != {k: v for k, v in old.items() if k != 'conditions'}:
                raise ValueError('Examples changed.')
            for method in METHODS:
                c = row['conditions'][method]
                check_cached_prediction(folder, config, row, c)
                if tokenizer.count(c['text']) != c['tokens'] or c['predicted_label'] != int(c['p_yes'] >= .5):
                    raise ValueError('Token count or predicted label mismatch.')
                if method != 'llm_summary':
                    if c != old['conditions'][method]:
                        raise ValueError('An unchanged condition was altered.')
                    continue
                entry = read_json(entry_path(folder, row['context'], method))
                source_entry = read_json(entry_path(SOURCE/name, row['context'], method))
                first = select_original_first(source_entry['candidates'])
                verify_summary_candidate(first, [], row['context'], c['budget_tokens'], config, tokenizer)
                if c['text'] != first['text'] or c['original_request_hash'] != first['request_hash']:
                    raise ValueError('Not the original first draft.')
                if c['budget_met'] != (c['tokens'] <= c['budget_tokens']) or c['post_clip_used'] or c['matching_trim_used'] or c['length_repair_used']:
                    raise ValueError('Incorrect budget reporting or edited first draft.')
                if entry['result_sha256'] != fingerprint(entry['result']):
                    raise ValueError('First draft output hash mismatch.')
                over += not c['budget_met']
        metrics = read_json(folder/'metrics.json')
        check_matrix(rows, metrics, METHODS)
        if metrics != analyze(rows, config):
            raise ValueError('Metrics replay differs.')
        write_json(folder/'audit.json', {'status': 'passed', 'n': 100, 'forecasts': 600, 'matrix_cells': 36,
                   'original_prompt_first_draft': True, 'unchanged_other_condition_forecasts': 500,
                   'summary_over_target_questions': over, 'summary_hard_cap_enforced': False,
                   'query_blind_inputs': True, 'no_post_truncation': True, 'new_compression_calls': 0,
                   'old_outputs_unchanged': True, 'reconstructed_requests': True})
    predict(config, tokenizer, offline=True)
    manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(SUITE/'manifest.json', manifest)
    print('First draft audit passed: 1000 unchanged controls, 1200 valid probabilities, 72 InfoGap cells.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'predict', 'audit', 'all'], default='all')
    args = parser.parse_args()
    config = read_json(CONFIG)
    tokenizer, manifest = prepare(config)
    if args.stage in {'predict', 'all'}:
        load_local_keys()
        predict(config, tokenizer)
    if args.stage in {'audit', 'all'}:
        audit(config, tokenizer, manifest)


if __name__ == '__main__':
    main()
