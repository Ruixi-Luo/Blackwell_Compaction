import inspect,unittest
from blackwell.summary_feedback import drafting_budget,feedback_messages
from blackwell.fixed_budget import in_band

class SummaryFeedbackTests(unittest.TestCase):
    def test_controller_corrects_a_systematic_writer_length_error(self):
        previous={'tokens':1200,'text':'draft','drafting_budget':1000}
        requested=drafting_budget(previous,1000)
        measured=round(1.2*requested)
        self.assertTrue(in_band(measured,1000))
    def test_controller_expands_short_output_without_changing_actual_cap(self):
        previous={'tokens':300,'text':'draft','drafting_budget':1000}
        self.assertGreater(drafting_budget(previous,1000),1000)
        self.assertLessEqual(drafting_budget(previous,1000),2000)
    def test_no_query_argument(self):
        self.assertEqual(list(inspect.signature(feedback_messages).parameters),['context','requested','previous','measured'])
        messages=feedback_messages('source',800,'old draft',1200)
        self.assertIn('800',messages[-1]['content'])
        self.assertIn('1200',messages[-1]['content'])

if __name__=='__main__':unittest.main()
