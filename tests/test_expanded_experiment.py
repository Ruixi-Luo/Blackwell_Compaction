"""新增数据边界及长度适配的回归测试；不调用网络或加载模型。"""
import unittest
from unittest.mock import patch
from blackwell.compressors import create_compressor
from blackwell.datasets import qasper, longbench
from blackwell.compressors.chunking import split_overlong
from blackwell.metrics import payoff_curve, infogap


def annotation(value, evidence):
    return {'answer': {'yes_no': value, 'evidence': evidence, 'unanswerable': False,
                       'extractive_spans': [], 'free_form_answer': ''}}


class ExpandedExperimentTests(unittest.TestCase):
    def test_device_is_explicitly_forwarded_to_the_selected_compressor(self):
        with patch('blackwell.compressors.llmlingua2.LLMLingua2Compressor') as factory:
            create_compressor('llmlingua2', device='cuda')
            factory.assert_called_once_with(device='cuda')

    def test_longbench_binary_rule_rejects_conflicts_and_mixed_answers(self):
        for answers, expected in [(['Yes.', ' YES '],1), (['no','No!'],0),
                                  (['yes','no'],None), (['yes','unanswerable'],None),
                                  ([],None), (['是的。'],None), (['Yes, it is.'],None)]:
            self.assertEqual(longbench.binary_label(answers), expected)

    def test_qasper_shared_paper_context_contains_neither_query_nor_gold(self):
        paper = {'title':'Title', 'abstract':'Abstract',
                 'full_text':[{'section_name':'Section','paragraphs':['Evidence.','Keep all paragraphs.']}],
                 'qas':[{'question_id':str(i),'question':f'PRIVATE_QUERY_{i}',
                         'answers':[annotation(value, ['Evidence.'])]} for i,value in enumerate([True,False])]}
        rows, excluded = qasper.convert_papers({'paper':paper},'train')
        self.assertEqual(len(rows),2)
        self.assertEqual(excluded,{})
        self.assertEqual([r['label'] for r in rows],[1,0])
        self.assertEqual(rows[0]['context_group'],rows[1]['context_group'])
        self.assertNotIn('PRIVATE_QUERY',rows[0]['context'])
        self.assertIn('Keep all paragraphs.',rows[0]['context'])
        paper['qas'][0]['answers'][0]['answer']['evidence'] = []
        filtered, excluded = qasper.convert_papers({'paper':paper},'train')
        self.assertEqual(len(filtered),1)
        self.assertEqual(excluded,{'missing_evidence':1})
        self.assertEqual(len(qasper.convert_papers({'paper':paper},'train',False)[0]),2)

    def test_length_split_preserves_unicode_whitespace_and_long_words(self):
        for text in ['A short sentence.', 'Long text. '*200, '界'*300, '  word\n\n'*100]:
            pieces = split_overlong(text, lambda s: len(s.encode('utf-8')), 40)
            self.assertEqual(''.join(pieces),text)
            self.assertTrue(all(len(p.encode('utf-8'))<=40 for p in pieces))
        self.assertEqual(split_overlong('unchanged',len,20),['unchanged'])

    def test_curve_extrema_match_both_directional_gaps_at_ties_and_endpoints(self):
        a,b,y=[0,.2,.9,1,.5],[1,.9,.2,0,.5],[0,1,1,0,1]
        curve=payoff_curve(a,b,y)
        self.assertAlmostEqual(max(0,*(p['payoff_gap'] for p in curve)),infogap(a,b,y)['value'])
        self.assertAlmostEqual(max(0,*(-p['payoff_gap'] for p in curve)),infogap(b,a,y)['value'])
        self.assertTrue(any(p['side']=='left_limit' for p in curve))


if __name__ == '__main__':
    unittest.main()
