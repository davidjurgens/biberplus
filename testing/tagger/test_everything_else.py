import unittest

import stanza

from src.tagger.tagger_utils import build_variable_dictionaries
from src.tagger.word_tagger import WordTagger


class TestAllOtherFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_whqu(self):
        doc = self.pipeline("only ended up as one due to Columbia Records . Why did it bomb ? Because it 's awful ,")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Why should be tagged as WHQU
        self.assertIn('WHQU', tagger.tagged_words[10]['tags'])

    def test_whcl(self):
        doc = self.pipeline("it gingerly with his foot . How could anyone know what to do with an assortment "
                            "like that ? Perhaps")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # What should be tagged as WHCL
        self.assertIn('WHCL', tagger.tagged_words[10]['tags'])

    def test_pin(self):
        doc = self.pipeline("have kept my hair in great condition ! A waste of money that 's all I have to say about")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Of should be tagged as PIN
        self.assertIn('PIN', tagger.tagged_words[10]['tags'])

    def test_conj(self):
        doc = self.pipeline("tips are a great feature . the wires are slick instead of the iPod 's slightly "
                            "grippy wires , which")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Instead should be tagged as CONJ
        self.assertIn('CONJ', tagger.tagged_words[10]['tags'])

    def test_demo(self):
        doc = self.pipeline("a little bigger than i expected . I just purchased this item and I "
                            "have not found anywhere on the")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # This should be tagged as DEMO
        self.assertIn('DEMO', tagger.tagged_words[10]['tags'])

    def test_cont(self):
        doc = self.pipeline("easy to get refund from that vendor if you do n't like the product "
                            "-LRB- same refund policy as with")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # n't should be tagged as CONT
        self.assertIn('CONT', tagger.tagged_words[10]['tags'])

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

    def test_peas(self):
        doc = self.pipeline("the exchange , so far all my dealings with amazon have been easy and fair . "
                            "Waterloo Station is a")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Have should be tagged as PEAS
        self.assertIn('PEAS', tagger.tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
