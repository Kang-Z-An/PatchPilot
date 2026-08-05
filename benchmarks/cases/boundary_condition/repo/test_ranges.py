import unittest

from ranges import inclusive_range


class InclusiveRangeTest(unittest.TestCase):
    def test_multiple_values(self) -> None:
        self.assertEqual(inclusive_range(2, 4), [2, 3, 4])

    def test_single_value(self) -> None:
        self.assertEqual(inclusive_range(3, 3), [3])

    def test_reversed_bounds(self) -> None:
        self.assertEqual(inclusive_range(4, 2), [])


if __name__ == "__main__":
    unittest.main()
