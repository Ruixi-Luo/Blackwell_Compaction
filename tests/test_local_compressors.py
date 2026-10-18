import inspect
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from blackwell.compressors import validate_input
from blackwell.compressors.llmlingua2 import LLMLingua2Compressor
from blackwell.compressors.selective_context import SelectiveContextCompressor, length_safe_sentences
from blackwell.local_cache import LocalCompressionCache
from blackwell.storage import ROOT, read_json, write_json
from run_experiment import selected_methods


class LocalCompressorTests(unittest.TestCase):
    def test_baseline_import_does_not_load_optional_ml_libraries(self):
        code = 'import sys; import blackwell.compression; assert "torch" not in sys.modules; assert "spacy" not in sys.modules'
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)

    def test_native_ratios_have_opposite_meanings_and_no_query_parameter(self):
        sc = SelectiveContextCompressor.__new__(SelectiveContextCompressor)
        sc.engine = Mock(return_value=('kept text', ['deleted']))
        sc.engine.beautify_context.return_value = 'one sentence'
        sc.engine.sent_tokenize_pattern = r'\n'
        sc.engine.tokenizer.encode.return_value = [1, 2, 3]
        sc.engine.max_token_length = 1024
        self.assertEqual(sc.compress('context', .25), 'kept text')
        sc.engine.assert_called_once_with('context', reduce_ratio=.75, reduce_level='phrase')
        llm = LLMLingua2Compressor.__new__(LLMLingua2Compressor)
        llm.engine = Mock()
        llm.engine.compress_prompt_llmlingua2.return_value = {'compressed_prompt': 'kept text'}
        self.assertEqual(llm.compress('context', .25), 'kept text')
        keywords = llm.engine.compress_prompt_llmlingua2.call_args.kwargs
        self.assertEqual(keywords['context'], ['context'])
        self.assertEqual(keywords['rate'], .25)
        self.assertNotIn('question', keywords)
        for cls in (SelectiveContextCompressor, LLMLingua2Compressor):
            self.assertEqual(list(inspect.signature(cls.compress).parameters), ['self', 'context', 'retained_ratio'])
        with self.assertRaises(TypeError):
            llm.compress('context', question='secret')

    def test_long_selective_context_sentence_is_split_without_losing_words(self):
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda text, **kwargs: [0] + text.removeprefix('<|endoftext|>').split()
        sentences = ['short sentence', 'word ' * 2100]
        parts = length_safe_sentences(sentences, tokenizer, 1024)
        self.assertEqual(parts[0], 'short sentence')
        self.assertEqual(' '.join(parts).split(), ' '.join(sentences).split())
        self.assertTrue(all(len(tokenizer.encode('<|endoftext|>'+p))<=1024 for p in parts))

    def test_local_cache_replays_without_loading_models_and_detects_changes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as temporary:
            run = Path(temporary)
            engine = Mock()
            engine.compress.return_value = 'kept'
            with patch('blackwell.local_cache.create_compressor', return_value=engine):
                result = LocalCompressionCache(run, {'revision': 'pinned'}).compress('context', 3, 'llmlingua2', .5)
            with patch('blackwell.local_cache.create_compressor', side_effect=AssertionError('No model loading')):
                replay = LocalCompressionCache(run, {'revision': 'pinned'}, offline=True)
                self.assertEqual(result, replay.compress('context', 3, 'llmlingua2', .5))
                with self.assertRaises(RuntimeError):
                    replay.compress('changed context', 3, 'llmlingua2', .5)
                with self.assertRaises(RuntimeError):
                    LocalCompressionCache(run, {'revision': 'other'}, offline=True).compress('context', 3, 'llmlingua2', .5)
            path = next((run / 'local_compression_cache').glob('*.json'))
            entry = read_json(path)
            entry['result']['text'] = 'tampered'
            write_json(path, entry)
            with self.assertRaises(ValueError):
                replay.compress('context', 3, 'llmlingua2', .5)

    def test_methods_and_context_inputs_are_validated(self):
        self.assertEqual(len(selected_methods(read_json(ROOT / 'configs/local_comparison.json'))), 6)
        for methods in [['full'], ['full','no_context','full'], ['full','no_context','typo']]:
            with self.assertRaises(ValueError):
                selected_methods({'methods': methods})
        for value in [0, -1, 1.1, True]:
            with self.assertRaises(ValueError):
                validate_input('context', value)
        with self.assertRaises(TypeError):
            validate_input({'context':'c','question':'q'}, .5)


if __name__ == '__main__':
    unittest.main()
