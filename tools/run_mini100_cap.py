"""Fixed 50% cap study: import verified caches -> predict -> audit.

No compression model is called. All prior directories remain read-only.
Only the first complete feasible summary is selected, before reading labels.
"""
import argparse
import math
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
from blackwell.cap_budget import first_complete_feasible
from blackwell.fixed_budget import in_band, revision_messages
from blackwell.summary_feedback import drafting_budget, feedback_messages
from blackwell.compressors.llm_rewrite import rewrite_messages
from blackwell.cost_guard import CostGuard, GuardedClient
from blackwell.diagnostics import probability_diagnostics
from run_experiment import load_local_keys, analyze

METHODS = ['full', 'no_context', 'selective_context', 'llmlingua2', 'llm_summary', 'dac']
SOURCES = ['hotpotqa', 'qasper']
SUITE = ROOT / 'runs/mini100_cap50_20261025_v4'
SOURCE = ROOT / 'runs/mini100_fixed50_20261025_v3'
DAC_SOURCE = ROOT / 'runs/mini100_fixed50_20261025'
CONFIG = ROOT / 'configs/mini100_cap50.json'


def core_hashes():
    paths = [Path(__file__), CONFIG, ROOT/'run_experiment.py', ROOT/'configs/dac.lock.json',
             *sorted((ROOT/'blackwell').rglob('*.py'))]
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}


def entry_path(folder, context, method):
    return folder/'compressions'/f'{fingerprint({"context": context, "method": method})}.json'


def payload(config, messages, maximum, json_mode=False):
    body = {'model': config['model'], 'messages': messages, 'temperature': config['temperature'],
            'max_tokens': maximum, 'enable_thinking': config['enable_thinking'],
            'stream': config.get('stream', False)}
    if json_mode:
        body['response_format'] = {'type': 'json_object'}
    return {'base_url': config['base_url'], 'payload': body}


def sources_index():
    return {p.stem: p for p in sorted((ROOT/'runs').rglob('api_cache/*.json')) if SUITE not in p.parents}


def verify_summary_candidate(candidate, prior, context, budget, config, tokenizer):
    """Rebuild the full source-only request and verify the complete raw response."""
    if tokenizer.count(candidate['text']) != candidate['tokens']:
        raise ValueError('Summary token count mismatch.')
    if 'origin_file' in candidate:
        origin_path = ROOT/candidate['origin_file']
        if file_hash(origin_path) != candidate['origin_sha256']:
            raise ValueError('Original summary cache changed.')
        origin = read_json(origin_path)
        if origin['request']['context'] != context or origin['request']['budget'] != budget:
            raise ValueError('Source context/budget mismatch.')
        original_candidates = origin['result']['candidates']
        index = next(i for i, c in enumerate(original_candidates)
                     if c['request_hash'] == candidate['request_hash'] and c['text'] == candidate['text'])
        previous = original_candidates[index-1] if index else None
        messages = rewrite_messages(context, budget, 'llm_summary',
                                    previous['text'] if previous else None,
                                    previous['tokens'] if previous else None)
        maximum = candidate.get('output_limit_retry', {}).get('retry_max_tokens', min(32768, max(512, 2*budget+512)))
        folder = origin_path.parent.parent/origin['result']['compression_api_subdir']
    else:
        if not prior:
            raise ValueError('Revision lacks its previous draft.')
        previous = prior[-1]
        if candidate.get('feedback_version') == 'proportional':
            if candidate['drafting_budget'] != drafting_budget(previous, budget):
                raise ValueError('Historical feedback setting mismatch.')
            messages = feedback_messages(context, candidate['drafting_budget'], previous['text'], previous['tokens'])
        else:
            messages = revision_messages(context, budget, previous['text'], previous['tokens'])
        maximum = candidate['max_tokens']
        folder = ROOT/candidate['api_folder']
    record_path = folder/'api_cache'/f"{candidate['request_hash']}.json"
    record = read_json(record_path)
    # Preserve the actual historical streaming/non-streaming request identity.
    cfg = {**config, 'model': config['compression_model'], 'stream': record['request']['payload']['stream']}
    expected = payload(cfg, messages, maximum)
    if record['request'] != expected or fingerprint(expected) != candidate['request_hash']:
        raise ValueError('Historical summary request mismatch.')
    text, _ = CachedChatClient._extract(record['response'], candidate['request_hash'])
    if text != candidate['text']:
        raise ValueError('Historical summary response mismatch.')
    return {**candidate, 'complete_verified': True, 'api_folder': folder.relative_to(ROOT).as_posix(),
            'raw_response_sha256': file_hash(record_path)}


def prepare(config):
    tokenizer = ContextTokenizer()
    if (SUITE/'manifest.json').exists():
        manifest = read_json(SUITE/'manifest.json')
        if manifest['config'] != config or manifest['core_hashes'] != core_hashes():
            raise ValueError('Frozen code/config changed; use a new identity.')
        return tokenizer, manifest
    source_manifest = read_json(SOURCE/'manifest.json')
    old_files = dict(source_manifest['old_files_sha256'])
    # Snapshot every file of the stopped v3 suite, including raw attempts/caches.
    for path in SOURCE.rglob('*'):
        if path.is_file():
            old_files[path.relative_to(ROOT).as_posix()] = file_hash(path)
    receipts = []
    for name in SOURCES:
        folder = SUITE/name
        examples = read_json(SOURCE/name/'examples.json')
        write_json(folder/'examples.json', examples)
        write_json(folder/'config.json', {**config, 'dataset': name})
        contexts = list(dict.fromkeys(e['context'] for e in examples))
        for context in contexts:
            budget = math.floor(config['retained_ratio']*tokenizer.count(context))
            for method in METHODS[2:]:
                source = entry_path(DAC_SOURCE/name if method == 'dac' else SOURCE/name, context, method)
                if method == 'llm_summary':
                    source = source.with_suffix('.drafts.json')
                    drafts = read_json(source)
                    candidates = [verify_summary_candidate(c, drafts[:i], context, budget, config, tokenizer)
                                  for i, c in enumerate(drafts)]
                    selected = first_complete_feasible(candidates, budget)
                    index = candidates.index(selected)
                    result = {'text': selected['text'], 'tokens': selected['tokens'], 'budget_tokens': budget,
                              'budget_met': True, 'budget_fill': selected['tokens']/budget,
                              'within_target_band': in_band(selected['tokens'], budget),
                              'post_clip_used': False, 'matching_trim_used': False,
                              'length_repair_used': index > 0, 'selected_candidate': index,
                              'selection_policy': 'first_complete_feasible', 'compression_seconds': 0.,
                              'compression_seconds_note': 'Cache reuse; historical generation costs recorded separately.',
                              'compression_request_hashes': [c['request_hash'] for c in candidates[:index+1]],
                              'historical_candidates_available': len(candidates), 'new_compression_calls': 0}
                    entry = {'request': {'context': context, 'budget_tokens': budget, 'method': method},
                             'candidates': candidates, 'result': result, 'result_sha256': fingerprint(result)}
                else:
                    entry = read_json(source)
                    if entry['request'] != {'context': context, 'budget_tokens': budget, 'method': method}:
                        raise ValueError('Local cache request mismatch.')
                    if entry['result_sha256'] != fingerprint(entry['result']):
                        raise ValueError('Local cache hash mismatch.')
                    if entry['result']['tokens'] > budget or entry['result']['text'] not in [c['text'] for c in entry['candidates']]:
                        raise ValueError('Local cache exceeds cap or uses post-truncation.')
                destination = entry_path(folder, context, method)
                write_json(destination, entry)
                old_files[source.relative_to(ROOT).as_posix()] = file_hash(source)
                receipts.append({'source': source.relative_to(ROOT).as_posix(), 'source_sha256': file_hash(source),
                                 'destination': destination.relative_to(ROOT).as_posix(), 'destination_sha256': file_hash(destination)})
    spending = read_json(SOURCE/'spending_reservations.json')
    for item in spending:
        if item['status'] == 'reserved':
            item['status'] = 'interrupted_usage_unknown'
            item['note'] = 'User stopped summary revisions. Full outstanding reservation retained; not assumed free.'
    write_json(SUITE/'spending_reservations.json', spending)
    shutil.copy2(DAC_SOURCE/'dac_preflight.json', SUITE/'dac_preflight.json')
    write_json(SUITE/'cache_reuse_provenance.json', {'files': receipts, 'new_compression_calls': 0,
               'summary_rule': 'First verified complete feasible candidate in frozen saved generation order.',
               'cost_inherited_once_from': (SOURCE/'spending_reservations.json').relative_to(ROOT).as_posix()})
    manifest = {'config': config, 'core_hashes': core_hashes(), 'old_files_sha256': old_files,
                'datasets': source_manifest['datasets'], 'status': 'prepared',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'Fixed 50% cap only; summary first complete feasible; other compressor outputs unchanged.',
                'source_runs': source_manifest['source_runs']}
    write_json(SUITE/'manifest.json', manifest)
    print('Prepared 197 contexts, 788 compression outputs, zero new compressor calls.', flush=True)
    return tokenizer, manifest


def read_condition(folder, context, method, tokenizer):
    if method in {'full', 'no_context'}:
        text = context if method == 'full' else ''
        return {'text': text, 'tokens': tokenizer.count(text), 'budget_tokens': None, 'budget_met': None,
                'length_repair_used': False, 'compression_request_hashes': []}
    entry = read_json(entry_path(folder, context, method))
    if entry['result_sha256'] != fingerprint(entry['result']):
        raise ValueError('Compression output hash mismatch.')
    return {**entry['result'], 'local_compression_hash': fingerprint(entry['request'])}


def audit(tokenizer, config, manifest):
    from tools.extend_with_salt import check_cached_prediction, check_matrix
    for path, digest in manifest['old_files_sha256'].items():
        if file_hash(ROOT/path) != digest:
            raise ValueError('Historical file changed: '+path)
    receipts = read_json(SUITE/'cache_reuse_provenance.json')['files']
    for receipt in receipts:
        if file_hash(ROOT/receipt['source']) != receipt['source_sha256'] or file_hash(ROOT/receipt['destination']) != receipt['destination_sha256']:
            raise ValueError('Compression import changed.')
    for name in SOURCES:
        folder = SUITE/name
        rows, examples = read_json(folder/'predictions.json'), read_json(folder/'examples.json')
        if len(rows) != 100 or fingerprint(examples) != manifest['datasets'][name]['examples_sha256']:
            raise ValueError('Sample mismatch.')
        checked = set()
        below_reference = {m: 0 for m in METHODS[2:]}
        for row, example in zip(rows, examples):
            if any(row[k] != v for k, v in example.items()) or list(row['conditions']) != METHODS:
                raise ValueError('Pairing mismatch.')
            for method, condition in row['conditions'].items():
                check_cached_prediction(folder, config, row, condition)
                if tokenizer.count(condition['text']) != condition['tokens'] or condition['predicted_label'] != int(condition['p_yes'] >= .5):
                    raise ValueError('Output count/label mismatch.')
                if method in METHODS[:2]:
                    if condition['text'] != (row['context'] if method == 'full' else ''):
                        raise ValueError('Control context changed.')
                    continue
                budget = math.floor(config['retained_ratio']*row['original_tokens'])
                if condition['budget_tokens'] != budget or condition['tokens'] > budget or condition['post_clip_used'] or condition['matching_trim_used']:
                    raise ValueError('Fixed cap violated.')
                below_reference[method] += not in_band(condition['tokens'], budget)
                path = entry_path(folder, row['context'], method)
                if path in checked:
                    continue
                checked.add(path)
                entry = read_json(path)
                if entry['request'] != {'context': row['context'], 'budget_tokens': budget, 'method': method}:
                    raise ValueError('Unexpected compressor inputs.')
                if method == 'llm_summary':
                    for index, candidate in enumerate(entry['candidates']):
                        verify_summary_candidate(candidate, entry['candidates'][:index], row['context'], budget, config, tokenizer)
                    selected = first_complete_feasible(entry['candidates'], budget)
                    if selected['text'] != condition['text'] or entry['candidates'].index(selected) != condition['selected_candidate']:
                        raise ValueError('First feasible selection mismatch.')
        metrics = read_json(folder/'metrics.json')
        check_matrix(rows, metrics, METHODS)
        if metrics != analyze(rows, {**config, 'sample_size': 100}):
            raise ValueError('Metrics replay mismatch.')
        write_json(folder/'audit.json', {'status': 'passed', 'n': 100, 'forecasts': 600, 'matrix_cells': 36,
            'query_blind_inputs': True, 'fixed_cap_passed': True, 'no_post_truncation': True,
            'summary_first_complete_feasible': True, 'new_compression_calls': 0,
            'below_47_5_percent_reference_counts': below_reference, 'minimum_length_required': False,
            'old_outputs_unchanged': True, 'reconstructed_requests': True})
    run_predictions(tokenizer, config, None, offline=True)
    manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(SUITE/'manifest.json', manifest)
    print('Audit passed: 1200 forecasts replayed, 72 matrix cells independently checked.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'predict', 'audit', 'all'], default='all')
    args = parser.parse_args()
    config = read_json(CONFIG)
    tokenizer, manifest = prepare(config)
    if args.stage in {'predict', 'all'}:
        load_local_keys()
        guard = CostGuard(SUITE/'spending_reservations.json', tokenizer, config['cost_reserve_cny'])
        run_predictions(tokenizer, config, guard)
    if args.stage in {'audit', 'all'}:
        audit(tokenizer, config, manifest)


def run_predictions(tokenizer,config,guard,offline=False):
    index=sources_index() if not offline else {}
    for name in SOURCES:
        folder=SUITE/name; examples=read_json(folder/'examples.json')
        cfg={**config,'stream':False,'timeout_seconds':90,'max_api_attempts':800,'min_interval_seconds':2.}
        client=CachedChatClient(cfg,folder,offline=True) if offline else GuardedClient(cfg,folder,guard)
        receipts=read_json(folder/'reader_reuse.json') if (folder/'reader_reuse.json').exists() else {}
        rows=[]
        for i,e in enumerate(examples,1):
            conditions={}; original=tokenizer.count(e['context'])
            for method in METHODS:
                c=read_condition(folder,e['context'],method,tokenizer)
                messages=prediction_messages(c['text'],e['question'])
                identity=payload(cfg,messages,128,True); key=fingerprint(identity)
                dest=folder/'api_cache'/f'{key}.json'
                if not offline and not dest.exists() and key in index:
                    source=index[key]; record=read_json(source)
                    if record['request']!=identity: raise ValueError('Reader cache identity mismatch')
                    parse_probability(CachedChatClient._extract(record['response'],key)[0])
                    dest.parent.mkdir(exist_ok=True); shutil.copy2(source,dest)
                    receipts[key]={'source':source.relative_to(ROOT).as_posix(),'sha256':file_hash(source)}
                raw,key=client.complete(messages,128,True); p=parse_probability(raw)
                conditions[method]={**c,'p_yes':p,'predicted_label':int(p>=.5),
                    'prediction_request_hash':key,'retained_ratio_actual':c['tokens']/original,
                    'reader_reused':key in receipts}
            rows.append({**e,'original_tokens':original,'conditions':conditions})
            if not offline:
                write_json(folder/'reader_reuse.json',receipts); write_json(folder/'predictions.json',rows)
                print(f'Predict {name} {i}/100',flush=True)
        if not offline:
            metrics=analyze(rows,{**config,'sample_size':100}); write_json(folder/'metrics.json',metrics)
            diagnostics={m:probability_diagnostics([r['conditions'][m]['p_yes'] for r in rows],[r['label'] for r in rows]) for m in METHODS}
            write_json(folder/'diagnostics.json',diagnostics)
        else:
            if rows!=read_json(folder/'predictions.json'): raise ValueError('Offline predictions differ')


if __name__ == "__main__":
    main()
