import unittest

import spacy

from biberplus.tagger import tag_text


class TestReducedFormsDispreferredStructuresFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_stpr(self):
        text = "plus a clicking noise each time you zoom in or out . My other complaints are that it does n't"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Out should be tagged as STPR
        self.assertIn('STPR', tagged_words[10]['tags'])

    def test_spin(self):
        text = "When all is said and done , this film seeks to financially cash in on the Rap\/Hip Hop culture and"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # To should be tagged as SPIN
        self.assertIn('SPIN', tagged_words[10]['tags'])

    def test_spin_to(self):
        text = "It's really hard to quickly understand some concepts, especially when trying to accurately capture details."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # To (before quickly understand) should be tagged as SPIN
        self.assertIn('SPIN', tagged_words[4]['tags'])

    def test_spau(self):
        text = "portray her three narrators in distinct fashions so that we can easily follow when one stops " \
               "and another begins ."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Are should be tagged as SPAU
        self.assertIn('SPAU', tagged_words[10]['tags'])

    def test_spau_with_single_adverb(self):
        text = "portray her three narrators in distinct fashions so that we can easily follow when one stops " \
               "and another begins ."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Can should be tagged as SPAU due to the presence of the adverb "easily" followed by the verb "follow".
        self.assertIn('SPAU', tagged_words[10]['tags'])

    def test_spau_with_double_adverbs(self):
        text = "She might quickly run to the store or she could slowly walk there."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Might should be tagged as SPAU due to the presence of the adverb "quickly" followed by the verb "run".
        self.assertIn('SPAU', tagged_words[1]['tags'])

    def test_thatd(self):
        text = "passes away and his wealth is gone ? Overall I thought this was a good book , it was n't"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Though should be tagged as THATD
        self.assertIn('THATD', tagged_words[10]['tags'])

    def test_thatd_with_demp_or_subject_ppronoun(self):
        text = "I said she should leave now."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Said should be tagged as THATD
        self.assertIn('THATD', tagged_words[1]['tags'])

    def test_thatd_with_modifier_noun_verb_pattern(self):
        text = "He suggests a different approach might work."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Suggests should be tagged as THATD
        self.assertIn('THATD', tagged_words[1]['tags'])

    def test_cont_n_t(self):
        text = "I can't believe it's true."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # "can't" should be tagged as CONT
        self.assertIn('CONT', tagged_words[2]['tags'])

    def test_cont_apostrophe(self):
        text = "They're going to the '80s themed party."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # "They're" should be tagged as CONT
        self.assertIn('CONT', tagged_words[1]['tags'])


if __name__ == '__main__':
    unittest.main()
