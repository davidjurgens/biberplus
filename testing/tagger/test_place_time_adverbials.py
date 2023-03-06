import unittest

import stanza

from src.tagger.word_tagger import WordTagger
from src.tagger.tagger_utils import build_variable_dictionaries


class TestPlaceTimeAdverbialFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_place(self):
        doc = self.pipeline('some phases and results that challenge thought as to how far patents are to'
                            ' become weapons of warfare in business')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Far should be tagged as a PLACE
        self.assertIn('PLACE', tagger.tagged_words[10]['tags'])

    def test_time(self):
        doc = self.pipeline('so the thermostat opens , and then look for leaks again . '
                            'Be sure the bugs and dirt are blown')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Again should be tagged as a TIME
        self.assertIn('TIME', tagger.tagged_words[10]['tags'])

    def test_caus(self):
        doc = self.pipeline(
            'they did also fall under the power of death , because they did eat in disobedience ; and disobedience to')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Because should be tagged as a CAUS
        self.assertIn('CAUS', tagger.tagged_words[10]['tags'])

    def test_conc(self):
        doc = self.pipeline(
            "outsider . When they learn you 're in the hills though , they 'll rally , do n't worry about")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Though should be tagged as a CONC
        self.assertIn('CONC', tagger.tagged_words[10]['tags'])

    def test_cond(self):
        doc = self.pipeline('so high that the top falls gently over , as if to show that it really is hair and not')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # If should be tagged as a COND
        self.assertIn('COND', tagger.tagged_words[10]['tags'])

    def test_osub(self):
        doc = self.pipeline(
            'his comment on the planter dynasties as they have existed since the decades before the Civil War . It may')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Since should be tagged as an OSUB
        self.assertIn('OSUB', tagger.tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
