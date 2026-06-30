"""Deterministic confirmation that the post-hoc BIN_ correction == retagging.

This is the hard gate before any already-tagged data is rewritten. It proves two
things:

1. Equivalence (config-independent): an embedded copy of the legacy buggy
   aggregation, run on the same freshly tagged tokens as the fixed pipeline,
   yields exactly half the binary mean. So doubling a stored (buggy) value
   reproduces what retagging with the fix would produce.
2. Real-sample fingerprint: a committed sample of real tagged records satisfies
   the bug invariants, and the correction utility recovers the expected values.
"""
import functools
import json
import math
import operator
import os
import unittest
from collections import defaultdict
from math import ceil

import numpy as np
import pandas as pd

from biberplus.tagger import (
    calculate_tag_frequencies,
    load_config,
    load_pipeline,
    tag_text,
)
from biberplus.tagger.constants import BIBER_PLUS_TAGS
from biberplus.tagger.tag_frequencies import (
    calculate_descriptive_stats,
    calculate_mean_word_length,
    calculate_type_token_ratio,
)
from biberplus.tagger.tagger_utils import build_variable_dictionaries
from tools.fix_binary_means import (
    correct_binary_vector,
    correct_features,
    correct_means_vector,
    validate_binary_features,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_binary_tagged.jsonl")

# A short single-window sample and a longer one we force into multiple windows.
SINGLE_WINDOW_TEXT = "The report was written by the committee and the data were analyzed."
MULTI_WINDOW_TEXT = (
    "The committee released its annual report on Tuesday. I really think these "
    "results are clear, and the authors have argued that the data should be "
    "reexamined. Why does this matter? Because readers want to know whether the "
    "conclusions can be trusted. We will probably revisit the topic again soon, "
    "since several reviewers requested additional passive constructions be tested."
)


# --- Embedded copy of the legacy (buggy) aggregation, for equivalence checks ---

def _legacy_load_tags(config):
    tags = []
    if config["biber"]:
        tags.extend(BIBER_PLUS_TAGS)
    if config["binary_tags"]:
        tags.extend(["BIN_" + tag for tag in BIBER_PLUS_TAGS])
    if config["function_words"]:
        fw = config["function_words_list"] or build_variable_dictionaries()["function_words"]
        tags.extend(fw)
    return tags


def _legacy_update_tag_counts(tagged_df, tag_counts, tags, tag_binary, weight=1.0):
    curr = pd.Series(
        functools.reduce(operator.iconcat, tagged_df.tags, []), dtype=pd.StringDtype()
    ).value_counts().to_dict()
    for tag in tags:
        count = round(curr[tag] * weight) if tag in curr else 0
        tag_counts[tag].append(count)
        if tag_binary and tag in BIBER_PLUS_TAGS:
            name = "BIN_" + tag if tag[:4] != "BIN_" else tag
            tag_counts[name].append(int(tag in curr))
    tag_counts["AWL"].append(calculate_mean_word_length(tagged_df))
    tag_counts["TTR"].append(calculate_type_token_ratio(tagged_df))
    return tag_counts


def _legacy_frequencies(text, pipeline, config):
    tagged = tag_text(text, pipeline, config)
    df = pd.DataFrame(tagged)
    tag_counts = defaultdict(list)
    tags = _legacy_load_tags(config)
    num_batches = ceil(len(df) / config["token_normalization"])
    for i, idx in enumerate(np.array_split(np.arange(len(df)), num_batches)):
        batch = df.iloc[idx]
        last = i == num_batches - 1
        if last and len(batch) <= config["drop_last_batch_pct"] * config["token_normalization"]:
            break
        weight = config["token_normalization"] / len(batch) if last else 1.0
        _legacy_update_tag_counts(batch, tag_counts, tags, config["binary_tags"], weight)
    return calculate_descriptive_stats(tag_counts)


class TestBinaryCorrectionEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.config.update({"binary_tags": True, "function_words": False})
        cls.pipeline = load_pipeline(cls.config)

    def _assert_equivalence(self, text, token_normalization):
        config = load_config()
        config.update(
            {"binary_tags": True, "function_words": False, "token_normalization": token_normalization}
        )
        legacy = _legacy_frequencies(text, self.pipeline, config).set_index("tag")
        fixed = calculate_tag_frequencies(text, self.pipeline, config).set_index("tag")

        for tag in BIBER_PLUS_TAGS:
            bk = "BIN_" + tag
            self.assertIn(bk, fixed.index)
            # Doubling the buggy mean reproduces the corrected (retagged) mean.
            self.assertAlmostEqual(
                fixed.loc[bk, "mean"], 2.0 * legacy.loc[bk, "mean"], places=9,
                msg=f"{bk}: fixed mean != 2 * legacy mean",
            )
            # Corrected std is fully determined by the corrected mean.
            p = fixed.loc[bk, "mean"]
            self.assertAlmostEqual(
                fixed.loc[bk, "std"], math.sqrt(max(p * (1 - p), 0.0)), places=9,
                msg=f"{bk}: fixed std != sqrt(p(1-p))",
            )

    def test_single_window_equivalence(self):
        self._assert_equivalence(SINGLE_WINDOW_TEXT, token_normalization=1000)

    def test_multi_window_equivalence(self):
        # Small normalization forces several windows over the longer text.
        self._assert_equivalence(MULTI_WINDOW_TEXT, token_normalization=15)


class TestRealSampleFingerprint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE, encoding="utf-8") as f:
            cls.records = [json.loads(line) for line in f if line.strip()]

    def test_fixture_present(self):
        self.assertGreater(len(self.records), 0)

    def test_records_match_bug_fingerprint(self):
        for i, rec in enumerate(self.records):
            problems = validate_binary_features(rec["features"])
            self.assertEqual(problems, [], f"record {i} violates bug invariants: {problems[:3]}")

    def test_utility_recovers_expected_values(self):
        for rec in self.records:
            before = rec["features"]
            corrected = correct_features(before)
            for key, val in before.items():
                if key.startswith("BIN_") and key.endswith("_mean"):
                    self.assertAlmostEqual(corrected[key], 2.0 * val, places=9)
                    std_key = key[: -len("_mean")] + "_std"
                    p = corrected[key]
                    self.assertAlmostEqual(
                        corrected[std_key], math.sqrt(max(p * (1 - p), 0.0)), places=9
                    )


class TestCorrectionPrimitives(unittest.TestCase):
    def test_single_window_features(self):
        # Single-window present feature: buggy mean 0.5 -> corrected 1.0, std 0.
        feats = {"BIN_PASS_mean": 0.5, "BIN_PASS_std": 0.5, "BIN_QUAN_mean": 0.0, "BIN_QUAN_std": 0.0}
        out = correct_features(feats)
        self.assertAlmostEqual(out["BIN_PASS_mean"], 1.0)
        self.assertAlmostEqual(out["BIN_PASS_std"], 0.0)
        self.assertAlmostEqual(out["BIN_QUAN_mean"], 0.0)
        self.assertAlmostEqual(out["BIN_QUAN_std"], 0.0)

    def test_multi_window_features(self):
        # p_buggy = 0.25 -> p_true = 0.5, std = sqrt(0.25) = 0.5.
        feats = {"BIN_NN_mean": 0.25, "BIN_NN_std": math.sqrt(0.25 * 0.75)}
        out = correct_features(feats)
        self.assertAlmostEqual(out["BIN_NN_mean"], 0.5)
        self.assertAlmostEqual(out["BIN_NN_std"], 0.5)

    def test_rejects_non_bug_data(self):
        # A mean > 0.5 cannot come from this bug; refuse to transform.
        with self.assertRaises(ValueError):
            correct_features({"BIN_NN_mean": 0.9, "BIN_NN_std": 0.3})

    def test_binary_vector_roundtrip(self):
        # Flattened [mean, std, mean, std]; second feature present in all windows.
        vec = [0.25, math.sqrt(0.25 * 0.75), 0.5, 0.5]
        out = correct_binary_vector(vec)
        self.assertAlmostEqual(out[0], 0.5)
        self.assertAlmostEqual(out[1], 0.5)
        self.assertAlmostEqual(out[2], 1.0)
        self.assertAlmostEqual(out[3], 0.0)

    def test_means_only_vector(self):
        out = correct_means_vector([0.5, 0.0, 0.25, 0.1])
        self.assertEqual(out, [1.0, 0.0, 0.5, 0.2])
        with self.assertRaises(ValueError):
            correct_means_vector([0.9])


if __name__ == "__main__":
    unittest.main()
