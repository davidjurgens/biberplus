import unittest

import spacy

from src.tagger.tagger_main import run_tagger_on_string


class TestPlaceTimeAdverbialFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_place(self):
        text = 'some phases and results that challenge thought as to how far patents are to become weapons of ' \
               'warfare in business'
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # Far should be tagged as a PLACE
        self.assertIn('PLACE', tagged_words[10]['tags'])

    def test_time(self):
        text = 'so the thermostat opens , and then look for leaks again . Be sure the bugs and dirt are blown'
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # Again should be tagged as a TIME
        self.assertIn('TIME', tagged_words[10]['tags'])

    def test_caus(self):
        text = 'they did also fall under the power of death , because they did eat in disobedience ; and disobedience to'
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # Because should be tagged as a CAUS
        self.assertIn('CAUS', tagged_words[10]['tags'])

    def test_conc(self):
        text = "outsider . When they learn you 're in the hills though , they 'll rally , do n't worry about"
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # Though should be tagged as a CONC
        self.assertIn('CONC', tagged_words[10]['tags'])

    def test_cond(self):
        text = 'so high that the top falls gently over , as if to show that it really is hair and not'
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # If should be tagged as a COND
        self.assertIn('COND', tagged_words[10]['tags'])

    def test_osub(self):
        text = 'his comment on the planter dynasties as they have existed since the decades before the Civil' \
               ' War . It may'
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # Since should be tagged as an OSUB
        self.assertIn('OSUB', tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
