import unittest

import spacy

from src.tagger.tagger_main import run_tagger_on_string


class TestPassivesFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_pass(self):
        text = 'sound scape is great , I am hearing nuance that was never heard before - an otherwise ' \
               'perfect headphone for'
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # Was should be tagged as a PASS
        self.assertIn('PASS', tagged_words[10]['tags'])

    def test_bypa(self):
        text = 'well after the Egyptian Golden Years , after Egypt had been conquered by the Greeks ... ' \
               'a time during which'
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # onquered should be tagged as a BYPA
        self.assertIn('BYPA', tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
