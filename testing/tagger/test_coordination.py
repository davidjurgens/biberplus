import unittest

import spacy

from src.tagger.tagger_main import run_tagger_on_string


class TestCoordinationFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_phc(self):
        text = "I dont think it was as good as Suicide Notes and Butterfly Kisses . " \
               "They should try to be as"
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # And should be tagged as a PHC
        self.assertIn('PHC', tagged_words[11]['tags'])

    def test_andc(self):
        text = "to write a book that has already been written , and she fails in comparison . Spend your money on"
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # And should be tagged as a ANDC
        self.assertIn('ANDC', tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
