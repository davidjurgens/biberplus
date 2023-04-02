import unittest

import spacy

from src.tagger.tagger_main import tag_string


class TestModalsFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_pomd(self):
        text = ", we 'll have to get organized . The baby can have an early nap . Victoria , I want"
        tagged_words = tag_string(self.pipeline, text)
        # Can should be tagged as a POMD
        self.assertIn('POMD', tagged_words[10]['tags'])

    def test_nemd(self):
        text = "social values , it is clear that the educational profession must work for the values which " \
               "are characteristic of the"
        tagged_words = tag_string(self.pipeline, text)
        # Must should be tagged as a NEMD
        self.assertIn('NEMD', tagged_words[10]['tags'])

    def test_prmd(self):
        text = "holds for values as well as life styles . One would need to test this proposition carefully ; after all"
        tagged_words = tag_string(self.pipeline, text)
        # Would should be tagged as a PRMD
        self.assertIn('PRMD', tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
