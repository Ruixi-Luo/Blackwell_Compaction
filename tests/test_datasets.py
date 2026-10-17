import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from blackwell.data import load_dataset
from blackwell.datasets import hotpotqa, qasper
from blackwell.storage import ROOT, download, read_json


def hotpot_row(answer='yes', identifier='test'):
    return {'id': identifier, 'answer': answer, 'question': 'PRIVATE_QUESTION', 'type': 'comparison',
            'supporting_facts': {'title': ['GOLD_SENTINEL']},
            'context': {'title': [f'Title {i}' for i in range(10)],
                        'sentences': [[f'Paragraph {i}. ', 'Second sentence.'] for i in range(10)]}}


def annotation(label, **changes):
    return {'answer': {'unanswerable': False, 'extractive_spans': [], 'free_form_answer': '',
                       'yes_no': label, 'evidence': ['Body evidence.'], **changes}}


class DatasetTests(unittest.TestCase):
    def test_hotpot_keeps_all_paragraphs_and_only_native_binary_labels(self):
        rows = [hotpot_row(' yes ', 'a'), hotpot_row('no', 'b'), hotpot_row('Paris', 'c')]
        examples = hotpotqa.convert_rows(rows)
        self.assertEqual([e['label'] for e in examples], [1, 0])
        self.assertEqual([e['source_row_zero_based'] for e in examples], [0, 1])
        expected = '\n\n'.join(f'Title {i}\nParagraph {i}. Second sentence.' for i in range(10))
        self.assertEqual(examples[0]['context'], expected)
        self.assertNotIn('PRIVATE_QUESTION', expected)
        self.assertNotIn('GOLD_SENTINEL', expected)
        changed = copy.deepcopy(rows)
        changed[0]['question'] = 'DIFFERENT QUESTION'
        changed[0]['supporting_facts'] = {'title': ['Title 5']}
        self.assertEqual(hotpotqa.convert_rows(changed)[0]['context'], expected)

    def test_hotpot_rejects_missing_documents_and_duplicate_ids(self):
        row = hotpot_row()
        row['context']['title'].pop()
        with self.assertRaises(ValueError):
            hotpotqa.convert_rows([row])
        with self.assertRaises(ValueError):
            hotpotqa.convert_rows([hotpot_row(), hotpot_row()])

    def test_hotpot_keeps_shorter_supplied_contexts_without_padding(self):
        row = hotpot_row()
        row['context'] = {k: v[:2] for k, v in row['context'].items()}
        example = hotpotqa.convert_rows([row])[0]
        self.assertEqual(example['paragraph_count'], 2)
        self.assertNotIn('Title 2', example['context'])

    def test_hotpot_sampling_is_seeded_and_not_balanced(self):
        pool = [{'id': i} for i in range(20)]
        with patch.object(hotpotqa, 'load_pool', return_value=(pool, {})):
            a, _ = hotpotqa.load_hotpotqa('dev', 5, 42)
            b, _ = hotpotqa.load_hotpotqa('dev', 5, 42)
            self.assertEqual(a, b)
            self.assertEqual(len({e['id'] for e in a}), 5)
            with self.assertRaises(ValueError):
                hotpotqa.load_hotpotqa('dev', 21, 42)
            with self.assertRaises(ValueError):
                hotpotqa.load_hotpotqa('test', 5, 42)

    def test_qasper_no_is_not_null_and_official_precedence_is_preserved(self):
        self.assertEqual(qasper.classify_question({'answers': [annotation(False)]}), ('eligible', 0))
        for answer in [annotation(None), annotation(False, unanswerable=True),
                       annotation(True, extractive_spans=['text']), annotation(True, free_form_answer='text')]:
            self.assertEqual(qasper.classify_question({'answers': [answer]})[0], 'no_boolean')

    def test_qasper_conflicts_and_mixed_annotations_are_excluded(self):
        self.assertEqual(qasper.classify_question({'answers': [annotation(True), annotation(False)]})[0],
                         'conflicting_boolean_labels')
        self.assertEqual(qasper.classify_question({'answers': [annotation(True), annotation(None, unanswerable=True)]})[0],
                         'mixed_answer_types')
        self.assertEqual(qasper.classify_question({'answers': []})[0], 'no_annotations')

    def test_qasper_full_text_and_conservative_evidence_check(self):
        paper = {'title': 'Title', 'abstract': 'Abstract', 'full_text': [
            {'section_name': 'Intro', 'paragraphs': ['Body evidence.', 'Other text.']}],
            'qas': ['PRIVATE_QUESTION_AND_LABEL']}
        self.assertEqual(qasper.build_context(paper), 'Title\n\nAbstract\n\nIntro\n\nBody evidence.\n\nOther text.')
        for evidence, expected in [(['Body evidence.'], 'text_verified'), ([], 'missing_evidence'),
                                   (['FLOAT SELECTED: Table 1'], 'figure_or_table'),
                                   (['missing passage'], 'evidence_not_in_text')]:
            question = {'answers': [annotation(True, evidence=evidence)]}
            self.assertEqual(qasper.text_evidence_status(question, paper), expected)

    def test_offline_download_miss_never_uses_network(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as folder, patch('blackwell.storage.urlopen') as network:
            with self.assertRaises(RuntimeError):
                download('https://example.com/data', Path(folder)/'missing', offline=True)
            network.assert_not_called()

    def test_boolq_refactor_preserves_the_saved_sample(self):
        old = ROOT/'runs/boolq_dev30_sc_llmlingua2_r50_seed20261017/examples.json'
        if not old.exists():
            self.skipTest('Historical local run is not present in a fresh clone.')
        config = read_json(ROOT/'configs/local_comparison.json')
        examples, _ = load_dataset(config, offline=True)
        self.assertEqual(examples, read_json(old))

    def test_unsupported_dataset_cannot_silently_become_boolq(self):
        with self.assertRaises(ValueError):
            load_dataset({'dataset': 'qasper'}, offline=True)


if __name__ == '__main__':
    unittest.main()
