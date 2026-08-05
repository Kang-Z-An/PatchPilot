import unittest

from ports import parse_port


class ParsePortTest(unittest.TestCase):
    def test_numeric_string(self) -> None:
        self.assertEqual(parse_port("8080"), 8080)

    def test_non_numeric_string_has_stable_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "^port must be an integer$"):
            parse_port("abc")

    def test_out_of_range_port(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 65535"):
            parse_port(70000)


if __name__ == "__main__":
    unittest.main()
