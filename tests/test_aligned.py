"""长度选择、概率分箱、query 隔离和显式失败的最小契约测试。"""
import inspect
import unittest
from unittest.mock import patch

from blackwell.budgeting import align_local, select_candidate
from blackwell.compressors.llm_rewrite import LLMRewriteCompressor, rewrite_messages
from blackwell.diagnostics import probability_diagnostics


class WordTokenizer:
    def count(self, text):
        return len(text.split())

    def head_tail(self, text, budget):
        return ' '.join(text.split()[:budget])


class AlignmentTests(unittest.TestCase):
    def test_parallel_worker_caps_sum_to_global_cap(self):
        from blackwell.rewrite_client import PacedRewriteClient
        config = {'max_api_attempts':17}
        with patch('blackwell.rewrite_client.CachedChatClient') as client:
            for worker in range(6):
                PacedRewriteClient(config, 'unused', WordTokenizer(),6,worker)
            limits = [call.args[0]['max_api_attempts'] for call in client.call_args_list]
        self.assertEqual(sum(limits),17)
        self.assertEqual(max(limits)-min(limits),1)

    def test_rewrite_pacing_preserves_the_request(self):
        from blackwell.rewrite_client import PacedRewriteClient
        messages = [{'role':'user','content':'source text'}]
        with patch('blackwell.rewrite_client.CachedChatClient') as base, \
             patch('blackwell.rewrite_client.time.monotonic',return_value=0.), \
             patch('blackwell.rewrite_client.time.sleep') as sleep:
            client = PacedRewriteClient({'max_api_attempts':10,'compression_tokens_per_minute':6000},'unused',WordTokenizer(),2,0)
            client.complete(messages,64)
            client.complete(messages,64)
            self.assertAlmostEqual(sleep.call_args.args[0],60*166/3000)
            base.return_value.complete.assert_called_with(messages,64)

    def test_compression_model_does_not_change_reader(self):
        from tools.run_aligned_suite import compression_config
        config = {'model':'reader', 'compression_model':'rewriter', 'max_api_attempts':10,
                  'compression_max_api_attempts':20,'timeout_seconds':90}
        adapted = compression_config(config)
        self.assertEqual(config['model'],'reader')
        self.assertEqual(adapted['model'],'rewriter')
        self.assertEqual(adapted['max_api_attempts'],20)
        self.assertEqual(adapted['timeout_seconds'],600)

    def test_feasible_candidate_and_explicit_clipping(self):
        candidates = [{'text':'a '*n, 'tokens':n} for n in (3, 7, 4)]
        result = select_candidate(candidates, 5, WordTokenizer())
        self.assertEqual(result['tokens'], 4)
        self.assertFalse(result['post_clip_used'])
        result = select_candidate(candidates[1:2], 5, WordTokenizer())
        self.assertTrue(result['post_clip_used'])
        self.assertEqual(result['tokens'], 5)

    def test_native_rate_is_length_only(self):
        class Engine:
            def compress(self, context, retained_ratio):
                return ' '.join(context.split()[:int(100*retained_ratio)+10])
        result = align_local(Engine(), 'x '*100, 50, WordTokenizer())
        self.assertLessEqual(result['tokens'], 50)
        self.assertGreaterEqual(result['tokens'], 47)
        self.assertLessEqual(len(result['candidates']), 8)

    def test_tiny_overshoot_preferred_to_large_underfill(self):
        candidates = [{'text':'a '*n, 'tokens':n} for n in (148,117,173)]
        result = select_candidate(candidates,146,WordTokenizer())
        self.assertEqual(result['selected_candidate'],0)
        self.assertEqual(result['tokens'],146)
        self.assertTrue(result['post_clip_used'])

    def test_query_boundary(self):
        self.assertNotIn('question', inspect.signature(LLMRewriteCompressor.compress).parameters)
        self.assertNotIn('label', inspect.signature(rewrite_messages).parameters)
        for method in ('llm_summary','telegrapher'):
            messages = rewrite_messages('CONTEXT_SENTINEL', 100, method)
            self.assertIn('CONTEXT_SENTINEL', messages[-1]['content'])
            self.assertNotIn('EVALUATION_QUERY_SENTINEL', str(messages))
            self.assertNotIn('p_yes', str(messages))

    def test_rewrite_length_repairs_and_cache_hashes(self):
        class Client:
            def __init__(self):
                self.calls = []
            def complete(self, messages, max_tokens):
                self.calls.append(messages)
                n = (2, 7, 5)[len(self.calls)-1]
                return 'x '*n, str(len(self.calls))
        client = Client()
        result = LLMRewriteCompressor(client, WordTokenizer(), 'llm_summary').compress('SOURCE '*10, 5)
        self.assertEqual(result['tokens'],5)
        self.assertEqual(result['compression_request_hashes'], ['1','2','3'])
        self.assertIn('2 Qwen tokens', client.calls[1][-1]['content'])

    def test_api_failure_does_not_silently_skip_context(self):
        class Client:
            def complete(self, *args, **kwargs):
                raise RuntimeError('failure')
        with self.assertRaises(RuntimeError):
            LLMRewriteCompressor(Client(), WordTokenizer(), 'llm_summary').compress('source',1)

    def test_bins_include_one_and_keep_empty_bins(self):
        result = probability_diagnostics([0.,.5,1.],[1,0,1])
        self.assertEqual(sum(b['n'] for b in result['reliability']),3)
        self.assertEqual(result['reliability'][-1]['n'],1)
        self.assertEqual(result['wrong_zero_or_one'],1)
        self.assertEqual(result['p_half'],1)
        self.assertAlmostEqual(result['ece_equal_width_10'],.5)
        self.assertGreater(result['log_loss_sensitivity']['1e-12'], result['log_loss_sensitivity']['0.001'])


if __name__ == '__main__':
    unittest.main()
