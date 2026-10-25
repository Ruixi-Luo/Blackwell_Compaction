import unittest
from tools.extend_mini100_firstdraft import select_original_first


class FirstDraftTest(unittest.TestCase):
    def test_first_not_shortest_even_when_above_target(self):
        first = dict(text='long first draft', tokens=80, origin_file='source', complete_verified=True)
        later = dict(text='short revision', tokens=40, revision=1, complete_verified=True)
        self.assertIs(select_original_first([first, later]), first)

    def test_no_fallback_from_incomplete_first(self):
        with self.assertRaises(ValueError):
            select_original_first([dict(text='partial', complete_verified=False),
                                   dict(text='later', origin_file='source', complete_verified=True)])

    def test_no_output_ceiling_retry_or_revision(self):
        for extra in [{'output_limit_retry': {'retry_max_tokens': 1000}}, {'revision': 1}]:
            with self.assertRaises(ValueError):
                select_original_first([dict(origin_file='source', complete_verified=True, **extra)])


if __name__ == '__main__':
    unittest.main()
