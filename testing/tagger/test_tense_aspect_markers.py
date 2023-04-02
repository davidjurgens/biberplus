import unittest

import spacy

from src.tagger.tagger_main import tag_string


class TestTenseAspectFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_peas(self):
        text = "the exchange , so far all my dealings with amazon have been easy and fair . " \
               "Waterloo Station is a"
        tagged_words = tag_string(self.pipeline, text)
        # Have should be tagged as PEAS
        self.assertIn('PEAS', tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
