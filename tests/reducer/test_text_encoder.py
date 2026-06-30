"""Unit tests for encode_frequencies (name/value alignment + group rules)."""
import unittest

import pandas as pd

from biberplus.reducer import encode_frequencies


def _frequencies_df():
    """Synthetic frequencies DataFrame mixing biber, binary, doc, and FW rows."""
    rows = [
        {"tag": "NN", "mean": 0.123456789, "min_val": 0, "max_val": 1, "range": 1, "std": 0.5},
        {"tag": "JJ", "mean": 0.2, "min_val": 0, "max_val": 1, "range": 1, "std": 0.4},
        {"tag": "BIN_NN", "mean": 1.0, "min_val": 1, "max_val": 1, "range": 0, "std": 0.0},
        {"tag": "BIN_JJ", "mean": 0.5, "min_val": 0, "max_val": 1, "range": 1, "std": 0.5},
        {"tag": "AWL", "mean": 4.2, "min_val": 3, "max_val": 5, "range": 2, "std": 0.7},
        {"tag": "TTR", "mean": 0.8, "min_val": 0.7, "max_val": 0.9, "range": 0.2, "std": 0.1},
        {"tag": "the", "mean": 0.9, "min_val": 0, "max_val": 2, "range": 2, "std": 0.3},
        {"tag": "of", "mean": 0.6, "min_val": 0, "max_val": 1, "range": 1, "std": 0.2},
    ]
    return pd.DataFrame(rows)


class TestEncodeFrequencies(unittest.TestCase):
    def setUp(self):
        self.df = _frequencies_df()
        self.config = {"binary_tags": True, "function_words": True}

    def test_names_values_aligned(self):
        enc = encode_frequencies(self.df, self.config)
        for group in ("biber", "binary", "function_words"):
            self.assertEqual(len(enc[group]["names"]), len(enc[group]["values"]))

    def test_biber_group_uses_all_columns(self):
        enc = encode_frequencies(self.df, self.config)
        # biber group covers NN, JJ, AWL, TTR (4 tags) x 5 columns = 20.
        self.assertEqual(len(enc["biber"]["values"]), 4 * 5)
        self.assertIn("NN_mean", enc["biber"]["names"])
        self.assertIn("AWL_std", enc["biber"]["names"])

    def test_binary_group_uses_mean_std_only(self):
        enc = encode_frequencies(self.df, self.config)
        names = enc["binary"]["names"]
        # 2 binary tags x {mean, std} = 4 values.
        self.assertEqual(len(enc["binary"]["values"]), 2 * 2)
        self.assertTrue(all(n.endswith("_mean") or n.endswith("_std") for n in names))
        self.assertEqual(names, ["BIN_NN_mean", "BIN_NN_std", "BIN_JJ_mean", "BIN_JJ_std"])

    def test_function_words_excludes_biber_and_binary(self):
        enc = encode_frequencies(self.df, self.config)
        names = enc["function_words"]["names"]
        self.assertEqual(len(enc["function_words"]["values"]), 2 * 5)
        # Only the function-word rows survive; biber/binary/doc tags are excluded.
        self.assertTrue(all(n.startswith("the_") or n.startswith("of_") for n in names))
        self.assertFalse(any(n.startswith(("BIN_", "NN_", "JJ_", "AWL_", "TTR_")) for n in names))

    def test_round_to_respected(self):
        enc = encode_frequencies(self.df, self.config, round_to=2)
        idx = enc["biber"]["names"].index("NN_mean")
        self.assertEqual(enc["biber"]["values"][idx], 0.12)

    def test_groups_absent_when_flags_off(self):
        enc = encode_frequencies(self.df, {"binary_tags": False, "function_words": False})
        self.assertIn("biber", enc)
        self.assertNotIn("binary", enc)
        self.assertNotIn("function_words", enc)


if __name__ == "__main__":
    unittest.main()
