import unittest

import spacy

from biberplus.tagger import tag_text


class TestNegationFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = spacy.load(
            "en_core_web_sm", disable=["parser", "lemmatizer", "ner"]
        )

    def test_syne(self):
        text = (
            "I did n't even want to give it away . No Beatrix Potter appreciator should "
            "be exposed to this ;"
        )
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn("SYNE", tagged_words[10]["tags"])

    def test_syne_neith(self):
        text = "I have neither skill nor patience for this task."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Neither should be tagged as SYNE before skill
        self.assertIn("SYNE", tagged_words[2]["tags"])

    def test_syne_no(self):
        text = "No good deed goes unpunished."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # No should be tagged as SYNE before good
        self.assertIn("SYNE", tagged_words[0]["tags"])

    def test_xx0(self):
        text = ", and General Motors on the other . It is not a medieval mental quirk or an attitude `` unnourished"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Not should be tagged as a XXO
        self.assertIn("XX0", tagged_words[10]["tags"])

    def test_syne_neither(self):
        text = "Neither option was acceptable."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Neither should be tagged as SYNE
        self.assertIn("SYNE", tagged_words[0]["tags"])

    def test_syne_nor(self):
        text = "The food was neither hot nor cold."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Both neither and nor should be tagged as SYNE
        self.assertIn("SYNE", tagged_words[3]["tags"])
        self.assertIn("SYNE", tagged_words[5]["tags"])

    def test_syne_no_with_adjective_noun(self):
        text = "There were no happy children at the party."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # No should be tagged as SYNE when followed by adj+noun
        self.assertIn("SYNE", tagged_words[2]["tags"])

    def test_xx0_not(self):
        text = "The movie was not interesting."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Not should be tagged as XX0
        self.assertIn("XX0", tagged_words[3]["tags"])

    def test_xx0_contraction(self):
        text = "I don't like this movie."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # n't should be tagged as XX0
        self.assertIn("XX0", tagged_words[2]["tags"])

    def test_xx0_with_adverb(self):
        text = "The movie was definitely not interesting."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Not should be tagged as XX0 even with intervening adverb
        self.assertIn("XX0", tagged_words[4]["tags"])


if __name__ == "__main__":
    unittest.main()
