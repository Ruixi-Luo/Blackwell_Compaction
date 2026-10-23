"""SALT 的数据边界和原生预算行为；用替身验证，不下载模型或调用 API。"""
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from blackwell.compressors.salt_summary import SaltSummaryCompressor


class SaltSummaryTests(unittest.TestCase):
    def make_compressor(self):
        compressor = SaltSummaryCompressor.__new__(SaltSummaryCompressor)
        compressor.pipeline = Mock()
        compressor.tokenizer = object()
        compressor.model = object()
        compressor.device = "cpu"
        compressor.select = Mock(return_value=([
            SimpleNamespace(text="Later sentence.", sent_idx=2),
            SimpleNamespace(text="Earlier sentence.", sent_idx=0),
        ], {}))
        compressor.pipeline.prep_prose_sentences.return_value = (["sentence"], ["all"], 20, 10, False)
        compressor.pipeline.build_prose_sent_data.return_value = ([{"text": "context", "n_words": 20}], {}, set())
        return compressor

    def test_query_free_interface_and_explicit_native_cli_settings(self):
        compressor = self.make_compressor()
        self.assertEqual(compressor.compress("context", .5), "Earlier sentence. Later sentence.")
        self.assertEqual(list(inspect.signature(compressor.compress).parameters), ["context", "retained_ratio"])
        compressor.pipeline.prep_prose_sentences.assert_called_once_with("context", "", compressor.tokenizer, .5)
        keywords = compressor.select.call_args.kwargs
        for name in ["query_keywords", "query_embedding", "query_proper_nouns", "qtype"]:
            self.assertIsNone(keywords[name])
        self.assertEqual(keywords["pass1_budget_pct"], .5)
        self.assertEqual(keywords["intro_pct"], .03)
        self.assertEqual(keywords["neighbor_window"], 0)
        compressor.pipeline.enrich_query.assert_not_called()
        with self.assertRaises(TypeError):
            compressor.compress("context", .5, query="secret question")
        with self.assertRaises(TypeError):
            compressor.compress({"context": "c", "query": "secret"}, .5)

    def test_filter_empty_output_and_native_budget_are_not_replaced(self):
        compressor = self.make_compressor()
        compressor.pipeline.prep_prose_sentences.return_value = ([], [], 0, 0, False)
        self.assertEqual(compressor.compress("only filtered text", .5), "")
        compressor.select.assert_not_called()
        compressor = self.make_compressor()
        compressor.pipeline.prep_prose_sentences.return_value = (["sentence"], ["all"], 90, 45, False)
        self.assertEqual(compressor.compress("context", .5), "context")
        compressor.select.assert_not_called()

    def test_identity_and_empty_inputs_bypass_native_pipeline(self):
        compressor = self.make_compressor()
        self.assertEqual(compressor.compress("context", 1), "context")
        self.assertEqual(compressor.compress("   ", .5), "   ")
        compressor.pipeline.prep_prose_sentences.assert_not_called()


if __name__ == "__main__":
    unittest.main()
