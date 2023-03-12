import unittest

import spacy

from src.tagger.tagger_main import run_tagger_on_string


class TestPrepPhrasesAdjectiveAdverbFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_pin(self):
        text = "have kept my hair in great condition ! A waste of money that 's all I have to say about"
        tagged_words = run_tagger_on_string(self.pipeline, text)

        # Of should be tagged as PIN
        self.assertIn('PIN', tagged_words[10]['tags'])

    def test_pred(self):
        text = "and rambling . Yeah , these guys were profound and impressive when I was in the 8th grade , but"
        tagged_words = run_tagger_on_string(self.pipeline, text)

        # Impressive should be tagged as a PRED
        self.assertIn('PRED', tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
