import unittest
from blackwell.cap_budget import first_complete_feasible


class CapBudgetTest(unittest.TestCase):
    def test_order_not_longest(self):
        candidates = [dict(text='first', tokens=30, complete_verified=True),
                      dict(text='later', tokens=49, complete_verified=True)]
        self.assertIs(first_complete_feasible(candidates, 50), candidates[0])

    def test_rejects_partial_empty_and_over_cap(self):
        candidates = [dict(text='partial', tokens=10, complete_verified=False),
                      dict(text=' ', tokens=1, complete_verified=True),
                      dict(text='too long', tokens=51, complete_verified=True),
                      dict(text='fits', tokens=50, complete_verified=True)]
        self.assertIs(first_complete_feasible(candidates, 50), candidates[3])

    def test_no_minimum_fill(self):
        candidate = dict(text='short', tokens=1, complete_verified=True)
        self.assertEqual(first_complete_feasible([candidate], 50), candidate)

    def test_no_fallback(self):
        with self.assertRaises(ValueError):
            first_complete_feasible([dict(text='partial', tokens=20)], 50)


if __name__ == '__main__':
    unittest.main()
