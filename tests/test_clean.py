"""Clean workflow regression tests: boundaries, complete drafts, replay and freezing."""
import copy
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from blackwell.storage import read_json, write_json
from blackwell.client import CachedChatClient, OutputLimitError
from blackwell.compressors.llm_rewrite import rewrite_messages
from experiment.config import load_config
from experiment.summary import summary_messages, generate_summary
from experiment.compression import compress_one, select_local, local_candidates, save_compression, load_compression
from experiment import pipeline
from run import run_lock


class WordCounter:
    def count(self, text):
        return len(text.split())


class CleanTests(unittest.TestCase):
    def test_original_prompt_is_byte_identical(self):
        for context, budget in [('Source with <tags> and 世界.', 10), ('Long facts.', 9000)]:
            self.assertEqual(summary_messages(context, budget), rewrite_messages(context, budget, 'llm_summary'))

    def test_summary_keeps_first_over_target_without_revision(self):
        api = Mock()
        api.complete.return_value = ('one two three four five six', 'hash')
        result = compress_one('source facts', 3, 'llm_summary', WordCounter(), client=api)
        self.assertFalse(result['budget_met'])
        self.assertEqual(result['text'], 'one two three four five six')
        self.assertEqual(api.complete.call_count, 1)
        self.assertFalse(result['length_repair_used'])

    def test_incomplete_summary_is_not_retried_or_accepted(self):
        api = Mock()
        api.complete.side_effect = OutputLimitError('hash')
        with self.assertRaises(OutputLimitError):
            generate_summary('source', 3, api, WordCounter())
        self.assertEqual(api.complete.call_count, 1)

    def test_compression_boundary_rejects_whole_example(self):
        self.assertNotIn('question', inspect.signature(compress_one).parameters)
        with self.assertRaises(ValueError):
            compress_one({'context': 'source', 'question': 'leak'}, 3, 'dac', WordCounter())

    def test_no_feasible_local_candidate_fails_without_clipping(self):
        with self.assertRaises(ValueError):
            select_local([{'text':'too long','tokens':11}], 10)
        self.assertEqual(select_local([{'tokens':3},{'tokens':8}],10)['tokens'],8)

    def test_local_search_is_bounded_and_never_changes_source(self):
        engine=Mock()
        engine.compress.return_value='one two three four five'
        rows=local_candidates('unaltered original context',2,'dac',engine,WordCounter())
        self.assertEqual(len(rows),4)
        self.assertEqual(engine.compress.call_args_list[0].kwargs['retained_ratio'],.4875)
        self.assertTrue(all(c.args[0]=='unaltered original context' for c in engine.compress.call_args_list))

    def test_cache_rejects_changed_budget_or_text(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            save_compression(folder,'source','dac',3,{'text':'small','tokens':1})
            self.assertEqual(load_compression(folder,'source','dac',3)['text'],'small')
            with self.assertRaises(ValueError):
                load_compression(folder,'source','dac',4)

    def test_offline_api_miss_never_opens_network(self):
        cfg=load_config('configs/clean/mini100.json')['reader']
        with tempfile.TemporaryDirectory() as temp, patch('blackwell.client.urlopen') as network:
            client=CachedChatClient(cfg,Path(temp),offline=True)
            with self.assertRaises(RuntimeError):
                client.complete([{'role':'user','content':'source'}],128,True)
            network.assert_not_called()

    def test_offline_compression_miss_never_loads_engine(self):
        cfg=load_config('configs/clean/smoke.json')
        with tempfile.TemporaryDirectory() as temp, patch('experiment.pipeline.create_engine') as engine:
            folder=Path(temp)
            for name in cfg['datasets']:
                write_json(folder/name/'examples.json',[{'context':'one two three four'}])
            with self.assertRaises(RuntimeError):
                pipeline.compress(cfg,folder,WordCounter(),offline=True)
            engine.assert_not_called()

    def test_lock_blocks_second_writer_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            with run_lock(folder):
                with self.assertRaises(RuntimeError):
                    with run_lock(folder):
                        pass
            self.assertFalse((folder/'.run.lock').exists())

    def test_prepare_freezes_config_without_legacy_dependencies(self):
        cfg=load_config('configs/clean/smoke.json')
        examples=[{'id':'1','context':'one two','question':'Q','label':1}]
        with tempfile.TemporaryDirectory() as temp, patch('experiment.pipeline.ContextTokenizer',return_value=WordCounter()), patch('experiment.pipeline.load_examples',return_value=(examples,{})):
            run=Path(temp)
            pipeline.prepare(cfg,run,False)
            changed=copy.deepcopy(cfg);changed['retained_ratio']=.3
            with self.assertRaises(ValueError):
                pipeline.prepare(changed,run,False)

    def test_no_current_module_imports_historical_runner(self):
        import ast
        for path in Path('experiment').glob('*.py'):
            tree=ast.parse(path.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith(('tools','run_experiment')))


    def test_fresh_pipeline_then_offline_replay_without_history(self):
        import json
        from blackwell.storage import fingerprint
        from experiment.api import request_identity
        config=load_config('configs/clean/smoke.json')
        config['device']='cpu'
        examples=[{'id':str(i),'context':'alpha beta gamma delta epsilon zeta',
                   'context_group':'same-paper','question':f'Question {i}?','label':i} for i in (0,1)]
        class FakeClient:
            def __init__(self, folder, settings, role):
                self.folder,self.settings,self.role=folder,settings,role
            def complete(self,messages,max_tokens,json_mode=False):
                identity=request_identity(self.settings,messages,max_tokens,json_mode)
                key=fingerprint(identity)
                content=json.dumps({'p_yes':.6}) if json_mode else 'alpha beta gamma delta epsilon'
                response={'choices':[{'finish_reason':'stop','message':{'content':content}}]}
                write_json(self.folder/'api'/self.role/'api_cache'/f'{key}.json',{'request':identity,'response':response})
                return content,key
        def fake_client(cfg,role,folder,tokenizer,allow_api=False):
            return FakeClient(folder,cfg[role],role)
        engine=Mock();engine.compress.return_value='alpha beta'
        fake_sys=Mock();fake_sys.flags.hash_randomization=0;fake_sys.modules={}
        with tempfile.TemporaryDirectory() as temp, patch('experiment.pipeline.ContextTokenizer',return_value=WordCounter()), patch('experiment.pipeline.load_examples',return_value=(examples,{})), patch('experiment.pipeline.sys',fake_sys), patch('experiment.pipeline.create_engine',return_value=engine), patch('blackwell.compressors.runtime.verify_assets'), patch('experiment.pipeline.make_client',side_effect=fake_client):
            run=Path(temp)
            tokenizer,manifest=pipeline.prepare(config,run)
            pipeline.compress(config,run,tokenizer)
            pipeline.predict(config,run,tokenizer)
            pipeline.evaluate(config,run)
            pipeline.audit(config,run,tokenizer,manifest)
            self.assertFalse((run/'import_provenance.json').exists())
            for name in config['datasets']:
                self.assertEqual(read_json(run/name/'audit.json')['forecasts'],12)
                self.assertEqual(read_json(run/name/'audit.json')['summary_over_target'],2)
            # Replaying calls no model: saved text and probabilities suffice.
            with patch('experiment.pipeline.make_client',side_effect=lambda cfg,role,folder,tok,allow_api=False: CachedChatClient(cfg[role],folder/'api'/role,offline=True)), patch('experiment.pipeline.create_engine') as forbidden:
                pipeline.compress(config,run,tokenizer,offline=True)
                pipeline.predict(config,run,tokenizer)
                forbidden.assert_not_called()
            pipeline.evaluate(config,run)
            pipeline.audit(config,run,tokenizer,manifest)


if __name__=='__main__':
    unittest.main()
