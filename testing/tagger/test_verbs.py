import unittest

import stanza

from src.tagger.simple_word_tagger import SimpleWordTagger
from src.tagger.tagger_utils import build_variable_dictionaries


class TestVerbFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_vbd(self):
        pass

    def test_priv(self):
        doc = self.pipeline('I expected to see ten, but instead saw twelve')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Expected should be tagged as a priv
        self.assertIn('PRIV', tagger.tagged_words[1]['tags'])

    def test_pubv(self):
        doc = self.pipeline('The candidate conceded the election late last night')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Conceded should be tagged as a pubv
        self.assertIn('PUBV', tagger.tagged_words[2]['tags'])

    def test_suav(self):
        doc = self.pipeline('I proposed extending the deadline')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Proposed should be tagged as a suav
        self.assertIn('SUAV', tagger.tagged_words[1]['tags'])

    def test_bema(self):
        doc = self.pipeline(
            'have a little boy that likes tractors , you can be certain this will be entertaining to him ! ')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # 'Be' should be tagged as BEMA
        self.assertIn('BEMA', tagger.tagged_words[10]['tags'])

    def test_smp(self):
        doc = self.pipeline('edge of the bank . From the outside , it seemed no more than a low drumlin , a lump')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Seemed should be tagged as SMP
        self.assertIn('SMP', tagger.tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
