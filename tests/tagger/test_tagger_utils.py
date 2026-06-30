"""Tests for config copy-independence and variable-dictionary caching."""
import tempfile
import unittest
from pathlib import Path

from biberplus.tagger.tagger_utils import (
    build_variable_dictionaries,
    load_config,
    read_in_variables,
)


class TestLoadConfig(unittest.TestCase):
    def test_returns_independent_copies(self):
        a = load_config()
        a["token_normalization"] = 99999
        a["binary_tags"] = not a.get("binary_tags", False)
        b = load_config()
        self.assertNotEqual(b["token_normalization"], 99999)

    def test_nested_mutation_isolated(self):
        a = load_config()
        a["function_words_list"] = ["mutated"]
        b = load_config()
        self.assertNotEqual(b.get("function_words_list"), ["mutated"])


class TestBuildVariableDictionaries(unittest.TestCase):
    def test_cached_identity(self):
        first = build_variable_dictionaries()
        second = build_variable_dictionaries()
        self.assertIs(first, second)

    def test_contains_expected_keys(self):
        variables = build_variable_dictionaries()
        self.assertIn("function_words", variables)
        self.assertGreater(len(variables["function_words"]), 0)


class TestReadInVariables(unittest.TestCase):
    def test_utf8_and_blank_line_handling(self):
        with tempfile.TemporaryDirectory() as d:
            fp = Path(d) / "vars.txt"
            fp.write_text("café\nrésumé\n\nnaïve\n", encoding="utf-8")
            self.assertEqual(read_in_variables(fp), {"café", "résumé", "naïve"})


if __name__ == "__main__":
    unittest.main()
