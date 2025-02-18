import unittest
from unittest.mock import patch
import spacy
import pandas as pd

from biberplus.tagger.tag_frequencies import *
from biberplus.tagger.biber_plus_tagger import BiberPlusTagger


class TestLexicalSpecificityFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_calculate_mean_word_length(self):
        sample_data = {
            'text': ["apple", "banana", "cherry", "date", "fig"]
        }
        df = pd.DataFrame(sample_data)

        result = calculate_mean_word_length(df)
        assert result == 4.8

    def test_calculate_type_token_ratio_without_limit(self):
        sample_data = {
            'text': ["apple", "apple", "banana", "banana", "cherry", "date", "date", "fig"]
        }
        df = pd.DataFrame(sample_data)

        result = calculate_type_token_ratio(df, first_n=None)
        assert result == 5 / 8

    def test_calculate_type_token_ratio_with_limit(self):
        sample_data = {
            'text': ["apple", "apple", "banana", "banana", "cherry", "date", "date", "fig"]
        }
        df = pd.DataFrame(sample_data)

        result = calculate_type_token_ratio(df, first_n=6)
        assert result == 4 / 6

    def test_calculate_total_adverbs(self):
        sample_data = {
            'text': ["quickly", "runs", "slowly", "jumps"],
            'upos': ["ADV", "VERB", "ADV", "VERB"]
        }
        df = pd.DataFrame(sample_data)

        result = calculate_total_adverbs(df)
        assert result == 2

    from unittest.mock import patch

    def test_calculate_tag_frequencies(self):
        pass

if __name__ == '__main__':
    unittest.main()