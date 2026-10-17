"""不调用 API：用真实请求/响应核验 query 隔离、概率与 InfoGap。"""
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blackwell.client import parse_probability
from blackwell.prompts import compression_messages, prediction_messages
from blackwell.storage import fingerprint, read_json, write_json
from blackwell.tokenization import ContextTokenizer


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pilot.json')
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    run = ROOT / 'runs' / config['run_name']
    manifest = read_json(run / 'manifest.json')
    require(manifest['status'] == 'complete', 'Run is not complete.')
    require(manifest['identity']['config'] == config, 'Config differs from the recorded run.')
    examples = read_json(run / 'examples.json')
    require(fingerprint(examples) == manifest['identity']['examples_sha256'], 'Example fingerprint differs.')
    rows = read_json(run / 'predictions.json')
    require(len(rows) == len(examples) == config['sample_size'], 'Sample counts differ.')
    require((ROOT / 'data/tokenizer/qwen3-30b-a3b-instruct-2507.json').exists(), 'Local tokenizer is missing; run Prepare first.')
    tokenizer = ContextTokenizer()
    checked_requests = set()

    def cached_text(request_hash, messages):
        cached = read_json(run / 'api_cache' / f'{request_hash}.json')
        require(fingerprint(cached['request']) == request_hash, 'Request fingerprint differs.')
        request = cached['request']
        require(request['base_url'] == config['base_url'], 'Provider URL differs.')
        require(request['payload']['model'] == config['model'], 'Model differs.')
        require(request['payload']['messages'] == messages, 'Actual messages do not match the allowed inputs.')
        choice = cached['response']['choices'][0]
        require(choice['finish_reason'] == 'stop', 'A truncated response was used.')
        checked_requests.add(request_hash)
        return choice['message']['content'].strip()

    for row, example in zip(rows, examples):
        require(all(row[key] == value for key, value in example.items()), 'Example fields or order differ.')
        require(set(row['conditions']) == {'full','no_context','head_tail','summary'}, 'Missing condition.')
        require(row['conditions']['full']['text'] == example['context'], 'Full context was changed.')
        require(row['conditions']['no_context']['text'] == '', 'No-context control is not empty.')
        require(row['original_tokens'] == tokenizer.count(example['context']), 'Original token count differs.')
        for method, condition in row['conditions'].items():
            require(condition['tokens'] == tokenizer.count(condition['text']), 'Recorded token count differs.')
            messages = prediction_messages(condition['text'], example['question'])
            raw = cached_text(condition['prediction_request_hash'], messages)
            require(parse_probability(raw) == condition['p_yes'], 'Saved probability differs from actual response.')
            if method in {'head_tail','summary'}:
                require(condition['budget_met'] == (condition['tokens'] <= condition['budget_tokens']), 'Budget flag differs.')
        summary = row['conditions']['summary']
        require(1 <= len(summary['compression_request_hashes']) <= 2, 'Unexpected summary attempt count.')
        previous, previous_tokens = None, None
        for request_hash in summary['compression_request_hashes']:
            # 只允许 passage、预算与上一版摘要；不允许问题、标签或 reader 输出。
            messages = compression_messages(example['context'], summary['budget_tokens'], previous, previous_tokens)
            previous = cached_text(request_hash, messages)
            previous_tokens = tokenizer.count(previous)
        require(previous == summary['text'], 'Final summary differs from the saved API output.')

    # 独立算两动作的实际收益，核验每个 InfoGap 单元格，不调用 infogap()。
    metrics = read_json(run / 'metrics.json')
    methods = list(rows[0]['conditions'])
    for a in methods:
        for b in methods:
            thresholds = {0., 1., *(row['conditions'][m]['p_yes'] for row in rows for m in (a,b))}
            thresholds |= {math.nextafter(t, -math.inf) for t in thresholds if t > 0}
            advantages = []
            for t in thresholds:
                differences = []
                for row in rows:
                    y = row['label']
                    pa, pb = (row['conditions'][m]['p_yes'] for m in (a,b))
                    utility_a = t-y if pa <= t else y-t
                    utility_b = t-y if pb <= t else y-t
                    differences.append(utility_a - utility_b)
                advantages.append(sum(differences) / len(rows))
            expected = max(0., *advantages)
            require(abs(metrics['infogap'][a][b]['value'] - expected) < 1e-12, 'Independent decision-payoff check failed.')
    result = {'status':'passed', 'samples':len(rows), 'forecasts':4*len(rows),
              'unique_api_responses_checked':len(checked_requests), 'infogap_cells_independently_checked':16,
              'checks':['sample identities', 'actual query-blind compression messages', 'raw probabilities',
                        'actual token counts', 'two-action payoff verification'], 'new_api_calls':0}
    write_json(run / 'audit.json', result)
    print(result)


if __name__ == '__main__':
    main()
