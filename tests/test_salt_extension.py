"""追加实验的关键保障：旧值不变、只预测新方法、请求严格一致、断网重放。"""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from blackwell.client import CachedChatClient
from blackwell.metrics import infogap
from blackwell.prompts import prediction_messages
from blackwell.storage import ROOT, fingerprint, read_json, write_json
from tools.extend_with_salt import (METHOD, check_cached_prediction, check_config,
                                    check_matrix, predict_rows, verify_old_conditions)


class SaltExtensionTests(unittest.TestCase):
    def setUp(self):
        self.old_config = read_json(ROOT / 'configs/hotpotqa_full.json')
        self.config = {**self.old_config, 'source_run': self.old_config['run_name'],
                       'run_name': 'salt_test', 'sample_size': 1,
                       'methods': self.old_config['methods'] + [METHOD],
                       'max_api_attempts': 12, 'min_interval_seconds': 3}
        self.old = {'id': '1', 'context': 'Only this context can be compressed.',
                    'question': 'QUESTION_SECRET: is it true?', 'label': 1, 'context_group': '1',
                    'original_tokens': 36,
                    'conditions': {m: {'text': m, 'p_yes': .6} for m in self.old_config['methods']}}

    def test_only_method_and_operational_changes_allowed(self):
        check_config(self.old_config, self.config)
        for name, changed in [('model', 'other'), ('sample_seed', 22), ('retained_ratio', .3),
                              ('enable_thinking', True), ('temperature', .1), ('bootstrap_seed', 99)]:
            with self.subTest(setting=name), self.assertRaisesRegex(ValueError, 'unchanged'):
                check_config(self.old_config, {**self.config, name: changed})
        for changes in ({'methods': [METHOD] + self.old_config['methods']},
                        {'sample_size': self.old_config['sample_size'] + 1},
                        {'run_name': self.old_config['run_name']}, {'source_run': '../escape'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                check_config(self.old_config, {**self.config, **changes})

    def test_old_conditions_cannot_be_changed_reordered_or_dropped(self):
        row = {**self.old, 'conditions': {**self.old['conditions'], METHOD: {'p_yes': .8}}}
        self.assertEqual(verify_old_conditions([row], [self.old], self.old_config['methods']), 4)
        for field, value in [('id', 'changed'), ('question', 'changed')]:
            with self.assertRaisesRegex(ValueError, 'Sample/order'):
                verify_old_conditions([{**row, field: value}], [self.old], self.old_config['methods'])
        changed = copy.deepcopy(row)
        changed['conditions']['full']['p_yes'] = .1
        with self.assertRaisesRegex(ValueError, 'Reused condition'):
            verify_old_conditions([changed], [self.old], self.old_config['methods'])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            verify_old_conditions([], [self.old], self.old_config['methods'])

    def test_only_salt_calls_reader_and_local_compressor_has_no_query(self):
        old = copy.deepcopy(self.old)
        old['original_tokens'] = len(old['context'])
        before = copy.deepcopy(old)
        compressed = 'Only context.'
        local_calls = []
        api_calls = []

        class FakeTokenizer:
            count = staticmethod(len)

        class FakeLocalCache:
            def __init__(self, run, identity, offline=False):
                assert offline is True

            def compress(self, *args, **kwargs):
                local_calls.append((args, kwargs))
                return {'text': compressed, 'compression_seconds': .25,
                        'native_retained_ratio': .5, 'local_compression_hash': 'test'}

        class FakeClient:
            new_attempts = 1
            cache_hits = 0

            def __init__(self, *args):
                pass

            def complete(self, messages, max_tokens, json_mode=False):
                api_calls.append((messages, max_tokens, json_mode))
                return '{"p_yes": 0.75}', 'request'

        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            run = Path(tmp)
            write_json(run / 'local_compression_check.json', {'status': 'passed'})
            with patch('tools.extend_with_salt.LocalCompressionCache', FakeLocalCache), \
                 patch('tools.extend_with_salt.CachedChatClient', FakeClient), \
                 patch('blackwell.compression.LOCAL_METHODS', {METHOD}):
                rows, execution = predict_rows(self.config, run, FakeTokenizer(), [old],
                                               {'identity': {'local_compressors': {}}})
            self.assertEqual(old, before)
            self.assertEqual(verify_old_conditions(rows, [old], self.old_config['methods']), 4)
            self.assertEqual(local_calls, [((old['context'], len(old['context']) // 2, METHOD, .5), {})])
            self.assertEqual(api_calls, [(prediction_messages(compressed, old['question']), 128, True)])
            self.assertEqual(execution['new_api_attempts'], 1)
            self.assertEqual(execution['reused_forecasts'], 4)
            self.assertEqual(read_json(run / 'predictions.json'), rows)

    def test_cached_prediction_requires_the_complete_request_and_raw_probability(self):
        condition = {'text': 'summary', 'p_yes': .7}
        request = {'base_url': self.config['base_url'], 'payload': {
            'model': self.config['model'], 'messages': prediction_messages('summary', self.old['question']),
            'temperature': 0., 'max_tokens': 128, 'enable_thinking': False, 'stream': False,
            'response_format': {'type': 'json_object'}}}
        key = fingerprint(request)
        condition['prediction_request_hash'] = key
        response = {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"p_yes": 0.7}'}}]}
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            folder = Path(tmp)
            path = folder / 'api_cache' / f'{key}.json'
            write_json(path, {'request': request, 'response': response})
            self.assertEqual(check_cached_prediction(folder, self.config, self.old, condition), key)
            # 真正的离线客户端只能命中同一完整请求；不会发送网络请求。
            client = CachedChatClient(self.config, folder, offline=True)
            text, replay_key = client.complete(request['payload']['messages'], 128, True)
            self.assertEqual(replay_key, key)
            self.assertEqual(text, '{"p_yes": 0.7}')
            self.assertEqual(client.new_attempts, 0)
            with self.assertRaisesRegex(RuntimeError, 'missing a cached response'):
                client.complete(request['payload']['messages'], 129, True)
            with self.assertRaisesRegex(ValueError, 'request hash differs'):
                check_cached_prediction(folder, {**self.config, 'temperature': .5}, self.old, condition)
            with self.assertRaisesRegex(ValueError, 'Probability differs'):
                check_cached_prediction(folder, self.config, self.old, {**condition, 'p_yes': .8})
            request['payload']['temperature'] = .5
            write_json(path, {'request': request, 'response': response})
            with self.assertRaisesRegex(ValueError, 'Cached prediction request differs'):
                check_cached_prediction(folder, self.config, self.old, condition)

    def test_independent_matrix_check_includes_left_limits(self):
        rows = [{'label': 0, 'conditions': {'a': {'p_yes': .1}, 'b': {'p_yes': .9}}}]
        values = {'a': [.1], 'b': [.9]}
        metrics = {'infogap': {a: {b: infogap(values[a], values[b], [0]) for b in values} for a in values}}
        check_matrix(rows, metrics, ['a', 'b'])
        metrics['infogap']['a']['b']['value'] = 1.7
        with self.assertRaisesRegex(ValueError, 'payoff verification'):
            check_matrix(rows, metrics, ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
