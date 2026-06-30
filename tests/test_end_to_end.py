"""End-to-end smoke tests for the public biberplus pipeline.

These run a realistic sample document through the full public API
(`calculate_tag_frequencies`, `encode_text`) under several config variants and
assert on output structure rather than on individual token tags (the latter is
covered by tests/tagger/). They act as the regression gate while refactoring the
package.
"""
import inspect
import unittest

import numpy as np
import pandas as pd

from biberplus.tagger import (
    BiberPlusTagger,
    calculate_tag_frequencies,
    load_config,
    load_pipeline,
    tag_text,
)
from biberplus.tagger.biber_run_order import RUN_ORDER
from biberplus.tagger.constants import BIBER_PLUS_TAGS
from biberplus.tagger.tagger_utils import build_variable_dictionaries
from biberplus.reducer import encode_text

SAMPLE_TEXT = (
    "The committee released its annual report on Tuesday. "
    "I really think that these results are surprisingly clear, "
    "and the authors have argued that the data should be reexamined. "
    "Why does this matter? Because readers want to know whether the "
    "conclusions can be trusted. We will probably revisit the topic soon."
)

FREQ_COLUMNS = {"tag", "mean", "min_val", "max_val", "range", "std"}


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.pipeline = load_pipeline(cls.config)
        cls.patterns = build_variable_dictionaries()

    def _config(self, **overrides):
        config = load_config()
        config.update(overrides)
        return config

    def test_default_frequencies_structure(self):
        df = calculate_tag_frequencies(SAMPLE_TEXT, self.pipeline, self.config)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(set(df.columns), FREQ_COLUMNS)
        self.assertGreater(len(df), 0)
        # Every Biber tag should be represented exactly once.
        tags = set(df["tag"])
        for biber_tag in BIBER_PLUS_TAGS:
            self.assertIn(biber_tag, tags)
        # No NaNs anywhere in the numeric columns.
        self.assertFalse(df.drop(columns="tag").isna().any().any())

    def test_binary_tags_variant(self):
        config = self._config(binary_tags=True)
        df = calculate_tag_frequencies(SAMPLE_TEXT, self.pipeline, config)
        tags = set(df["tag"])
        self.assertIn("BIN_PASS", tags)
        # Binary columns are 0/1 valued.
        bin_rows = df[df["tag"].str.startswith("BIN_")]
        self.assertTrue(((bin_rows["mean"] >= 0) & (bin_rows["mean"] <= 1)).all())

    def test_function_words_toggle(self):
        with_fw = calculate_tag_frequencies(
            SAMPLE_TEXT, self.pipeline, self._config(function_words=True)
        )
        without_fw = calculate_tag_frequencies(
            SAMPLE_TEXT, self.pipeline, self._config(function_words=False)
        )
        self.assertGreater(len(with_fw), len(without_fw))

    def test_encode_text_structure(self):
        config = self._config(binary_tags=True, function_words=True)
        encodings = encode_text(config, SAMPLE_TEXT)
        self.assertEqual({"biber", "binary", "function_words"}, set(encodings))
        for group in encodings.values():
            self.assertEqual({"names", "values"}, set(group))
            # Names and values are positionally aligned.
            self.assertEqual(len(group["names"]), len(group["values"]))
            self.assertGreater(len(group["values"]), 0)
            self.assertTrue(all(isinstance(v, float) for v in group["values"]))
            self.assertTrue(all(isinstance(n, str) for n in group["names"]))

    def test_encode_text_names_align_with_frequencies(self):
        config = self._config(binary_tags=False, function_words=False)
        df = calculate_tag_frequencies(SAMPLE_TEXT, self.pipeline, config)
        encodings = encode_text(config, SAMPLE_TEXT)
        biber = encodings["biber"]
        # 5 value columns per tag, names suffixed by column.
        self.assertEqual(len(biber["values"]), 5 * len(df))
        self.assertTrue(biber["names"][0].endswith("_mean"))
        # The value for "<tag>_mean" matches the DataFrame's mean for that tag.
        name_to_value = dict(zip(biber["names"], biber["values"]))
        sample_tag = df.iloc[0]["tag"]
        self.assertAlmostEqual(
            name_to_value[f"{sample_tag}_mean"], round(df.iloc[0]["mean"], 10)
        )

    def test_tag_text_token_dicts(self):
        tagged = tag_text(SAMPLE_TEXT, pipeline=self.pipeline, config=self.config)
        self.assertEqual(len(tagged), len([w for w in tagged]))
        for word in tagged:
            self.assertEqual({"text", "upos", "xpos", "feats", "tags"}, set(word))
            self.assertIsInstance(word["tags"], list)

    def test_short_input_does_not_crash(self):
        """A minimal one-token input should tag without raising."""
        df = calculate_tag_frequencies("Hello.", self.pipeline, self.config)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(set(df.columns), FREQ_COLUMNS)

    def test_parallel_matches_serial(self):
        """Multiprocessing path must yield identical tags to the serial path."""
        # Small processing_size forces batching/parallel branches on a short text.
        long_text = " ".join([SAMPLE_TEXT] * 4)

        serial_cfg = self._config(processing_size=8, n_processes=1)
        parallel_cfg = self._config(processing_size=8, n_processes=2)

        serial = tag_text(long_text, config=serial_cfg)
        parallel = tag_text(long_text, config=parallel_cfg)

        self.assertEqual(len(serial), len(parallel))
        self.assertEqual(
            [(w["text"], w["tags"]) for w in serial],
            [(w["text"], w["tags"]) for w in parallel],
        )

    def test_run_order_covers_all_tag_methods(self):
        """Invariant: every tag_* method on BiberPlusTagger appears in RUN_ORDER."""
        tagger = BiberPlusTagger([], self.patterns)
        tag_methods = {
            name
            for name in dir(tagger)
            if name.startswith("tag")
            and inspect.ismethod(getattr(tagger, name))
        }
        missing = tag_methods - set(RUN_ORDER)
        self.assertEqual(missing, set(), f"tag methods missing from RUN_ORDER: {missing}")


if __name__ == "__main__":
    unittest.main()
