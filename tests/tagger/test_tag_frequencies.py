"""Regression tests for the BIN_ double-count fix and the binary aggregation.

The bug: BIN_ tags were appended twice per window ([indicator, 0]), halving the
binary mean and inflating std. These tests pin the corrected behavior.
"""
import math
import unittest
from collections import defaultdict

import numpy as np
import pandas as pd
import spacy

from biberplus.tagger import calculate_tag_frequencies, load_config
from biberplus.tagger.constants import BIBER_PLUS_TAGS
from biberplus.tagger.tag_frequencies import (
    calculate_descriptive_stats,
    calculate_mean_word_length,
    calculate_type_token_ratio,
    count_tags_every_n_tokens,
    update_tag_counts,
)


def _df(rows):
    """Build a synthetic tagged DataFrame from (text, [tags]) pairs."""
    return pd.DataFrame([{"text": text, "tags": tags} for text, tags in rows])


class TestBinaryTagFrequencies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "ner"])

    def _config(self, **overrides):
        config = load_config()
        config.update(overrides)
        return config

    def _freq(self, text, **overrides):
        config = self._config(binary_tags=True, function_words=False, **overrides)
        return calculate_tag_frequencies(text, self.pipeline, config).set_index("tag")

    def test_present_feature_single_window_is_one(self):
        # A clear passive in a single window: BIN_PASS should be fully present.
        df = self._freq("The report was written by the committee.")
        self.assertAlmostEqual(df.loc["BIN_PASS", "mean"], 1.0)
        self.assertAlmostEqual(df.loc["BIN_PASS", "min_val"], 1.0)
        self.assertAlmostEqual(df.loc["BIN_PASS", "max_val"], 1.0)
        self.assertAlmostEqual(df.loc["BIN_PASS", "std"], 0.0)

    def test_absent_feature_is_zero(self):
        df = self._freq("The report was written by the committee.")
        # Emoji never occurs in this text.
        self.assertAlmostEqual(df.loc["BIN_EMOJ", "mean"], 0.0)
        self.assertAlmostEqual(df.loc["BIN_EMOJ", "max_val"], 0.0)

    def test_binary_mean_is_fraction_of_windows(self):
        # Force several windows; BIN_ mean must be a valid fraction in [0, 1]
        # and std must equal sqrt(p(1-p)) for that fraction.
        text = (
            "The committee released its annual report on Tuesday. I really think "
            "these results are clear, and the authors have argued that the data "
            "should be reexamined. Why does this matter? Because readers want to "
            "know whether the conclusions can be trusted. We will revisit soon."
        )
        df = self._freq(text, token_normalization=15)
        for tag in ["BIN_NN", "BIN_PASS", "BIN_VPRT", "BIN_JJ"]:
            p = df.loc[tag, "mean"]
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
            self.assertAlmostEqual(df.loc[tag, "std"], math.sqrt(max(p * (1 - p), 0.0)), places=9)

    def test_binary_independent_of_biber_flag(self):
        # With biber disabled but binary on, BIN_ columns must still be produced
        # and correct (decoupled from the biber flag).
        df = self._freq("The report was written by the committee.", biber=False)
        self.assertIn("BIN_PASS", df.index)
        self.assertAlmostEqual(df.loc["BIN_PASS", "mean"], 1.0)


class TestUpdateTagCounts(unittest.TestCase):
    """Synthetic-DataFrame unit tests for update_tag_counts (no spaCy)."""

    def test_counts_and_absent_tags(self):
        df = _df([("cats", ["NN", "NN"]), ("run", ["VB"])])
        out = update_tag_counts(df, defaultdict(list), ["NN", "VB", "JJ"], tag_binary=False)
        self.assertEqual(out["NN"], [2])
        self.assertEqual(out["VB"], [1])
        self.assertEqual(out["JJ"], [0])

    def test_weight_scales_and_rounds(self):
        df = _df([("cats", ["NN", "NN"]), ("run", ["VB"])])
        out = update_tag_counts(df, defaultdict(list), ["NN", "VB"], tag_binary=False, weight=2.0)
        self.assertEqual(out["NN"], [4])
        self.assertEqual(out["VB"], [2])

    def test_binary_indicator_single_entry_per_window(self):
        df = _df([("cats", ["NN", "NN"]), ("red", ["JJ"])])
        out = update_tag_counts(df, defaultdict(list), ["NN", "JJ"], tag_binary=True)
        # Present features -> 1, absent -> 0, exactly one indicator per window.
        self.assertEqual(out["BIN_NN"], [1])
        self.assertEqual(out["BIN_JJ"], [1])
        absent = next(t for t in BIBER_PLUS_TAGS if t not in ("NN", "JJ"))
        self.assertEqual(out["BIN_" + absent], [0])
        self.assertEqual(len(out["BIN_NN"]), 1)

    def test_doc_level_tags_appended(self):
        df = _df([("a", ["NN"]), ("bbb", ["NN"])])
        out = update_tag_counts(df, defaultdict(list), ["NN"], tag_binary=False)
        self.assertAlmostEqual(out["AWL"][0], 2.0)  # (1 + 3) / 2
        self.assertAlmostEqual(out["TTR"][0], 1.0)  # both tokens unique


class TestWindowing(unittest.TestCase):
    def _config(self, **overrides):
        cfg = {"token_normalization": 2, "drop_last_batch_pct": 0.0, "binary_tags": False}
        cfg.update(overrides)
        return cfg

    def test_window_count(self):
        df = _df([(c, ["NN"]) for c in "abcd"])
        out = count_tags_every_n_tokens(df, defaultdict(list), ["NN"], self._config())
        self.assertEqual(len(out["NN"]), 2)  # 4 tokens / window of 2

    def test_drop_small_last_batch(self):
        df = _df([(c, ["NN"]) for c in "abc"])  # windows [a,b],[c]
        out = count_tags_every_n_tokens(df, defaultdict(list), ["NN"], self._config(drop_last_batch_pct=0.5))
        self.assertEqual(len(out["NN"]), 1)  # tiny last batch dropped

    def test_last_batch_weighting(self):
        df = _df([(c, ["NN"]) for c in "abc"])  # windows [a,b]=2, [c]=1
        out = count_tags_every_n_tokens(df, defaultdict(list), ["NN"], self._config())
        # last window scaled by token_normalization / len = 2 / 1 = 2
        self.assertEqual(out["NN"], [2, 2])


class TestDescriptiveStats(unittest.TestCase):
    def test_stats(self):
        stats = calculate_descriptive_stats({"NN": [1, 3]}).set_index("tag")
        self.assertAlmostEqual(stats.loc["NN", "mean"], 2.0)
        self.assertAlmostEqual(stats.loc["NN", "min_val"], 1)
        self.assertAlmostEqual(stats.loc["NN", "max_val"], 3)
        self.assertAlmostEqual(stats.loc["NN", "range"], 2)
        self.assertAlmostEqual(stats.loc["NN", "std"], np.std([1, 3]))


class TestTtrAndAwl(unittest.TestCase):
    def test_type_token_ratio(self):
        df = _df([("a", []), ("a", []), ("b", [])])
        self.assertAlmostEqual(calculate_type_token_ratio(df), 2 / 3)

    def test_type_token_ratio_first_n(self):
        df = _df([("a", []) for _ in range(500)])
        self.assertAlmostEqual(calculate_type_token_ratio(df, first_n=400), 1 / 400)

    def test_mean_word_length(self):
        df = _df([("a", []), ("bbb", [])])
        self.assertAlmostEqual(calculate_mean_word_length(df), 2.0)


if __name__ == "__main__":
    unittest.main()
