import unittest

import spacy

from src.tagger.tagger_main import run_tagger_on_string


class TestSpecializedVerbFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_priv(self):
        text = 'I expected to see ten, but instead saw twelve'
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # Expected should be tagged as a priv
        self.assertIn('PRIV', tagged_words[1]['tags'])

    def test_pubv(self):
        text = 'The candidate conceded the election late last night'
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # Conceded should be tagged as a pubv
        self.assertIn('PUBV', tagged_words[2]['tags'])

    def test_suav(self):
        text = 'I proposed extending the deadline'
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # Proposed should be tagged as a suav
        self.assertIn('SUAV', tagged_words[1]['tags'])

    def test_smp(self):
        text = 'edge of the bank . From the outside , it seemed no more than a low drumlin , a lump'
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # Seemed should be tagged as SMP
        self.assertIn('SMP', tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
