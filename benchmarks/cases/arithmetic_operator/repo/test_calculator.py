import unittest

from calculator import add


class CalculatorTest(unittest.TestCase):
    def test_positive_integers(self) -> None:
        self.assertEqual(add(2, 3), 5)

    def test_negative_integer(self) -> None:
        self.assertEqual(add(7, -2), 5)


if __name__ == "__main__":
    unittest.main()
