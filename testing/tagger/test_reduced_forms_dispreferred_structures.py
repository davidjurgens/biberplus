import unittest

import spacy

from src.tagger.tagger_utils import build_variable_dictionaries
from src.tagger.word_tagger import WordTagger


class TestReducedFormsDispreferredStructuresFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])
        self.patterns_dict = build_variable_dictionaries()

    def test_stpr(self):
        doc = self.pipeline("plus a clicking noise each time you zoom in or out . My other complaints "
                            "are that it does n't")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Out should be tagged as STPR
        self.assertIn('STPR', tagger.tagged_words[10]['tags'])

    def test_spin(self):
        doc = self.pipeline("When all is said and done , this film seeks to financially cash in "
                            "on the Rap\/Hip Hop culture and")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # To should be tagged as SPIN
        self.assertIn('SPIN', tagger.tagged_words[10]['tags'])

    def test_spau(self):
        doc = self.pipeline("portray her three narrators in distinct fashions so that we can easily follow "
                            "when one stops and another begins .")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Are should be tagged as SPAU
        self.assertIn('SPAU', tagger.tagged_words[10]['tags'])

    def test_thatd(self):
        doc = self.pipeline("passes away and his wealth is gone ? Overall I thought this was a good book , it was n't")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Though should be tagged as THATD
        self.assertIn('THATD', tagger.tagged_words[10]['tags'])

    def test_cont(self):
        doc = self.pipeline("easy to get refund from that vendor if you do n't like the product "
                            "-LRB- same refund policy as with")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # n't should be tagged as CONT
        self.assertIn('CONT', tagger.tagged_words[10]['tags'])

if __name__ == '__main__':
    unittest.main()
