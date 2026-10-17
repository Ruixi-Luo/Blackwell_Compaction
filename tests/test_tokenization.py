import unittest

from blackwell.storage import ROOT


@unittest.skipUnless((ROOT / "data" / "tokenizer" / "qwen3-30b-a3b-instruct-2507.json").exists(), "Run Prepare once to enable local-tokenizer checks.")
class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from blackwell.tokenization import ContextTokenizer
        cls.tokenizer = ContextTokenizer()

    def test_head_tail_obeys_budget_including_separator(self):
        for text in ["The beginning has an important fact. " * 30,
                     "开头的信息和结尾的信息都需要保留。😀 café " * 20]:
            for budget in [1,2,5,10,30,100]:
                result = self.tokenizer.head_tail(text, budget)
                self.assertLessEqual(self.tokenizer.count(result), budget)
                self.assertNotIn("\ufffd", result)

    def test_no_change_when_already_short_enough(self):
        self.assertEqual(self.tokenizer.head_tail("short text", 100), "short text")


if __name__ == "__main__":
    unittest.main()
