import unittest

import spacy

from bibermda.tagger import tag_string


class TestNominalFormFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_nomz(self):
        text = 'consular materials to reveal the motives which led the British government to permit Garibaldi to cross ' \
               'the Straits of Messina'
        tagged_words = tag_string(self.pipeline, text)
        # Government should be tagged as a NOMZ
        self.assertIn('NOMZ', tagged_words[10]['tags'])

    def test_ger(self):
        text = "His voice carries the album well even with some subpar songwriting . I do n't know where " \
               "people are getting"
        tagged_words = tag_string(self.pipeline, text)
        # Campaigning should be tagged as a GER
        self.assertIn('GER', tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
