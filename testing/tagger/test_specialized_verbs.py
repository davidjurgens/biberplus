import unittest

import spacy

from biberplus.tagger import tag_text


class TestSpecializedVerbFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_priv(self):
        text = 'I expected to see ten, but instead saw twelve'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Expected should be tagged as a priv
        self.assertIn('PRIV', tagged_words[1]['tags'])

    def test_pubv(self):
        text = 'The candidate conceded the election late last night'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Conceded should be tagged as a pubv
        self.assertIn('PUBV', tagged_words[2]['tags'])

    def test_suav(self):
        text = 'I proposed extending the deadline'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Proposed should be tagged as a suav
        self.assertIn('SUAV', tagged_words[1]['tags'])

    def test_smp(self):
        text = 'edge of the bank . From the outside , it seemed no more than a low drumlin , a lump'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Seemed should be tagged as SMP
        self.assertIn('SMP', tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
