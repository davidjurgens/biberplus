import unittest

import spacy

from bibermda.tagger import tag_text


class TestTenseAspectFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_peas(self):
        text = "the exchange , so far all my dealings with amazon have been easy and fair . " \
               "Waterloo Station is a"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Have should be tagged as PEAS
        self.assertIn('PEAS', tagged_words[10]['tags'])

    def test_peas_direct_verb(self):
        text = "They have visited the museum several times."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Have should be tagged as PEAS
        self.assertIn('PEAS', tagged_words[1]['tags'])

    def test_peas_one_intervening_word(self):
        text = "I have always loved that painting."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Have should be tagged as PEAS
        self.assertIn('PEAS', tagged_words[1]['tags'])

    def test_peas_two_intervening_adverbs(self):
        text = "She has probably never been to the opera."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Has should be tagged as PEAS
        self.assertIn('PEAS', tagged_words[1]['tags'])


if __name__ == '__main__':
    unittest.main()
