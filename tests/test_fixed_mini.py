import tempfile, unittest, math
from pathlib import Path
from blackwell.fixed_budget import in_band, choose_candidate, revision_messages
from blackwell.cost_guard import CostGuard

class FakeTokenizer:
    def count(self,text): return len(text.split())

class FixedBudgetTests(unittest.TestCase):
    def test_short_method_cannot_change_another_budget(self):
        a=choose_candidate([{'text':'long','tokens':98}],100)
        b=choose_candidate([{'text':'short','tokens':30}],100)
        self.assertEqual((a['tokens'],b['tokens']),(98,30))
    def test_overshoot_is_not_truncated(self):
        with self.assertRaises(ValueError): choose_candidate([{'text':'intact','tokens':101}],100)
    def test_selection_is_length_only(self):
        self.assertEqual(choose_candidate([{'text':'a','tokens':95},{'text':'b','tokens':99},{'text':'c','tokens':102}],100)['text'],'b')
    def test_revision_boundary(self):
        import inspect
        self.assertEqual(list(inspect.signature(revision_messages).parameters),['context','budget','previous','previous_tokens'])
        messages=revision_messages('unique source',100,'draft',40)
        self.assertIn('unique source',messages[1]['content'])
        self.assertIn('NO other word limit',messages[1]['content'])
    def test_guard_reservations_and_unknown_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard=CostGuard(Path(tmp)/'cost.json',FakeTokenizer(),.02)
            i=guard.reserve('DeepSeek',[{'content':'source'}],100)
            original=guard.items[i]['accounted_cny']
            guard.settle(i,[{'status':'failed'}])
            self.assertGreater(guard.items[i]['accounted_cny'],0)
            self.assertAlmostEqual(guard.items[i]['accounted_cny'],original/3)
            with self.assertRaises(RuntimeError): guard.reserve('DeepSeek',[{'content':'source'}],100000)
    def test_guard_reuse_no_new_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard=CostGuard(Path(tmp)/'cost.json',FakeTokenizer())
            i=guard.reserve('DeepSeek',[{'content':'source'}],100)
            guard.settle(i,[])
            self.assertEqual(guard.items[i]['accounted_cny'],0)

if __name__=='__main__': unittest.main()
