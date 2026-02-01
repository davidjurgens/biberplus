import unittest

import spacy

from biberplus.tagger import tag_text


class TestReducedFormsDispreferredStructuresFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = spacy.load(
            "en_core_web_sm", disable=["parser", "lemmatizer", "ner"]
        )

    def test_stpr(self):
        text = "plus a clicking noise each time you zoom in or out . My other complaints are that it does n't"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Out should be tagged as STPR
        self.assertIn("STPR", tagged_words[10]["tags"])

    def test_spin(self):
        text = "When all is said and done , this film seeks to financially cash in on the Rap\/Hip Hop culture and"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # To should be tagged as SPIN
        self.assertIn("SPIN", tagged_words[10]["tags"])

    def test_spin_to(self):
        text = "It's really hard to quickly understand some concepts, especially when trying to accurately capture details."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # To (before quickly understand) should be tagged as SPIN
        self.assertIn("SPIN", tagged_words[4]["tags"])

    def test_spau(self):
        text = (
            "portray her three narrators in distinct fashions so that we can easily follow when one stops "
            "and another begins ."
        )
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Are should be tagged as SPAU
        self.assertIn("SPAU", tagged_words[10]["tags"])

    def test_spau_with_single_adverb(self):
        text = (
            "portray her three narrators in distinct fashions so that we can easily follow when one stops "
            "and another begins ."
        )
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Can should be tagged as SPAU due to the presence of the adverb "easily" followed by the verb "follow".
        self.assertIn("SPAU", tagged_words[10]["tags"])

    def test_spau_with_double_adverbs(self):
        text = "She might quickly run to the store or she could slowly walk there."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Might should be tagged as SPAU due to the presence of the adverb "quickly" followed by the verb "run".
        self.assertIn("SPAU", tagged_words[1]["tags"])

    def test_thatd(self):
        text = "passes away and his wealth is gone ? Overall I thought this was a good book , it was n't"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Though should be tagged as THATD
        self.assertIn("THATD", tagged_words[10]["tags"])

    def test_thatd_with_demp_or_subject_ppronoun(self):
        text = "I said she should leave now."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Said should be tagged as THATD
        self.assertIn("THATD", tagged_words[1]["tags"])

    def test_thatd_with_modifier_noun_verb_pattern(self):
        text = "He suggests a different approach might work."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Suggests should be tagged as THATD
        self.assertIn("THATD", tagged_words[1]["tags"])

    def test_cont_n_t(self):
        text = "I can't believe it's true."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # "can't" should be tagged as CONT
        self.assertIn("CONT", tagged_words[2]["tags"])

    def test_cont_apostrophe(self):
        text = "They're going to the '80s themed party."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # "They're" should be tagged as CONT
        self.assertIn("CONT", tagged_words[1]["tags"])

    def test_stpr_multiple_cases(self):
        text = "That's the store I went to. This is what I was thinking of. Who did you talk with?"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Check each stranded preposition
        self.assertIn("STPR", tagged_words[6]["tags"])  # 'to'
        self.assertIn("STPR", tagged_words[14]["tags"])  # 'of'
        self.assertIn("STPR", tagged_words[20]["tags"])  # 'with'

    def test_spin_negative(self):
        text = "I want to explain this concept."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # 'to' should NOT be tagged as SPIN since there's no intervening adverb
        self.assertNotIn("SPIN", tagged_words[2]["tags"])

    def test_spau_modal(self):
        text = "You should carefully consider the options."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # 'should' should be tagged as SPAU
        self.assertIn("SPAU", tagged_words[1]["tags"])

    def test_spau_have(self):
        text = "They have thoroughly reviewed the document."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # 'have' should be tagged as SPAU
        self.assertIn("SPAU", tagged_words[1]["tags"])

    def test_spau_be(self):
        text = "The results are clearly shown in the graph."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # 'are' should be tagged as SPAU
        self.assertIn("SPAU", tagged_words[2]["tags"])

    def test_cont_possessive(self):
        text = "That's John's book. The cat's tail."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Test that possessive 's is not tagged as contraction
        self.assertNotIn("CONT", tagged_words[1]["tags"])  # "'s" from "That's"
        self.assertNotIn("CONT", tagged_words[3]["tags"])  # "'s" from "John's"
        self.assertNotIn("CONT", tagged_words[8]["tags"])  # "'s" from "cat's"


if __name__ == "__main__":
    unittest.main()
