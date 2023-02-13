import unittest

import stanza

from src.tagger.simple_word_tagger import SimpleWordTagger
from src.tagger.tagger_utils import build_variable_dictionaries


class TestSubordinatorsFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_caus(self):
        doc = self.pipeline(
            'they did also fall under the power of death , because they did eat in disobedience ; and disobedience to')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Because should be tagged as a CAUS
        self.assertIn('CAUS', tagger.tagged_words[10]['tags'])

    def test_conc(self):
        doc = self.pipeline(
            "outsider . When they learn you 're in the hills though , they 'll rally , do n't worry about")
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Though should be tagged as a CONC
        self.assertIn('CONC', tagger.tagged_words[10]['tags'])

    def test_cond(self):
        doc = self.pipeline('so high that the top falls gently over , as if to show that it really is hair and not')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # If should be tagged as a COND
        self.assertIn('COND', tagger.tagged_words[10]['tags'])

    def test_osub(self):
        doc = self.pipeline(
            'his comment on the planter dynasties as they have existed since the decades before the Civil War . It may')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Since should be tagged as an OSUB
        self.assertIn('OSUB', tagger.tagged_words[10]['tags'])

    def test_thatd(self):
        doc = self.pipeline('be as in depth in that area . I just found that this book helped tremendously '
                            'in regards to its')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Found should be tagged as THATD
        self.assertIn('THATD', tagger.tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
