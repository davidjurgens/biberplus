import unittest

import spacy

from src.tagger.tagger_main import run_tagger_on_string


class TestNegationFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_syne(self):
        text = "I did n't even want to give it away . No Beatrix Potter appreciator should " \
               "be exposed to this ;"
        tagged_words = run_tagger_on_string(self.pipeline, text)
        self.assertIn('SYNE', tagged_words[10]['tags'])

    def test_xx0(self):
        text = ', and General Motors on the other . It is not a medieval mental quirk or an attitude `` unnourished'
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # Not should be tagged as a XXO
        self.assertIn('XX0', tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
