import unittest

import spacy

from biberplus.tagger import tag_text


class TestCoordinationFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_phc(self):
        text = "I dont think it was as good as Suicide Notes and Butterfly Kisses . " \
               "They should try to be as"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # And should be tagged as a PHC
        self.assertIn('PHC', tagged_words[11]['tags'])

    def test_pred_direct_case(self):
        text = "The cake is delicious in every way."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # 'delicious' should be tagged as a PRED
        self.assertIn('PRED', tagged_words[3]['tags'])

    def test_pred_phrasal_coordinator(self):
        text = "The car is fast and reliable for long trips."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # 'reliable' should be tagged as a PRED
        self.assertIn('PRED', tagged_words[5]['tags'])

    def test_andc(self):
        text = "to write a book that has already been written , and she fails in comparison . Spend your money on"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # And should be tagged as a ANDC
        self.assertIn('ANDC', tagged_words[10]['tags'])

    def test_andc_preceded_by_punctuation(self):
        text = "She finished her chores; and she went to bed."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # 'and' should be tagged as ANDC since it's preceded by a semicolon
        self.assertIn('ANDC', tagged_words[5]['tags'])

    def test_andc_followed_by_wh_word(self):
        text = "They asked him about his preferences and why he made that choice."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # 'and' should be tagged as ANDC since it's followed by 'why' which is a WH word
        self.assertIn('ANDC', tagged_words[6]['tags'])

if __name__ == '__main__':
    unittest.main()
