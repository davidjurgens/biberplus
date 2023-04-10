import unittest

import spacy

from src.tagger.tagger import tag_string


class TestReducedFormsDispreferredStructuresFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_stpr(self):
        text = "plus a clicking noise each time you zoom in or out . My other complaints are that it does n't"
        tagged_words = tag_string(self.pipeline, text)
        # Out should be tagged as STPR
        self.assertIn('STPR', tagged_words[10]['tags'])

    def test_spin(self):
        text = "When all is said and done , this film seeks to financially cash in on the Rap\/Hip Hop culture and"
        tagged_words = tag_string(self.pipeline, text)
        # To should be tagged as SPIN
        self.assertIn('SPIN', tagged_words[10]['tags'])

    def test_spau(self):
        text = "portray her three narrators in distinct fashions so that we can easily follow when one stops " \
               "and another begins ."
        tagged_words = tag_string(self.pipeline, text)
        # Are should be tagged as SPAU
        self.assertIn('SPAU', tagged_words[10]['tags'])

    def test_thatd(self):
        text = "passes away and his wealth is gone ? Overall I thought this was a good book , it was n't"
        tagged_words = tag_string(self.pipeline, text)
        # Though should be tagged as THATD
        self.assertIn('THATD', tagged_words[10]['tags'])

    def test_cont(self):
        text = "easy to get refund from that vendor if you do n't like the product -LRB- same refund policy as with"
        tagged_words = tag_string(self.pipeline, text)
        # n't should be tagged as CONT
        self.assertIn('CONT', tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
