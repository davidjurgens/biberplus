import unittest


class TestPassiveFunctions(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_pass(self):
        self.assertEqual(True, False)  # add assertion here

    def test_bypa(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
