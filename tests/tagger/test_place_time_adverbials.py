import unittest

import spacy

from biberplus.tagger import tag_text


class TestPlaceTimeAdverbialFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = spacy.load(
            "en_core_web_sm", disable=["parser", "lemmatizer", "ner"]
        )

    def test_place_adverbials(self):
        # Test basic place adverbials
        text1 = "The ship sailed abroad on its maiden voyage."
        tagged_words1 = tag_text(text1, pipeline=self.pipeline)
        self.assertIn("PLACE", tagged_words1[3]["tags"])

        # Test directional adverbials
        text2 = "The hikers went upstream to find the source."
        tagged_words2 = tag_text(text2, pipeline=self.pipeline)
        self.assertIn("PLACE", tagged_words2[3]["tags"])

        # Test indoor/outdoor locations
        text3 = "The children played outdoors while it was sunny."
        tagged_words3 = tag_text(text3, pipeline=self.pipeline)
        self.assertIn("PLACE", tagged_words3[3]["tags"])

        # Test cardinal directions
        text4 = "Birds fly south for the winter."
        tagged_words4 = tag_text(text4, pipeline=self.pipeline)
        self.assertIn("PLACE", tagged_words4[2]["tags"])

        # Test relative positions
        text5 = "The cat sat underneath the table."
        tagged_words5 = tag_text(text5, pipeline=self.pipeline)
        self.assertIn("PLACE", tagged_words5[3]["tags"])

    def test_place_adverbial_exceptions(self):
        # Test proper noun exception (e.g., North America should not be tagged)
        text = "They visited North America last summer."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertNotIn("PLACE", tagged_words[2]["tags"])

    def test_time_adverbials(self):
        # Test basic time adverbials
        text1 = "I will see you tomorrow at noon."
        tagged_words1 = tag_text(text1, pipeline=self.pipeline)
        self.assertIn("TIME", tagged_words1[4]["tags"])

        # Test relative time markers
        text2 = "He had previously visited this place."
        tagged_words2 = tag_text(text2, pipeline=self.pipeline)
        self.assertIn("TIME", tagged_words2[2]["tags"])

        # Test immediate time markers
        text3 = "The response came instantly."
        tagged_words3 = tag_text(text3, pipeline=self.pipeline)
        self.assertIn("TIME", tagged_words3[3]["tags"])

        # Test sequential time markers
        text4 = "First eat dinner, then do homework, and afterwards watch TV."
        tagged_words4 = tag_text(text4, pipeline=self.pipeline)
        self.assertIn("TIME", tagged_words4[9]["tags"])

    def test_time_adverbial_exceptions(self):
        # Test 'soon as' exception
        text = "As soon as possible, we will start."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertNotIn("TIME", tagged_words[2]["tags"])

    def test_multiple_adverbials(self):
        # Test combination of place and time adverbials
        text = "We went abroad yesterday to explore."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn("PLACE", tagged_words[2]["tags"])
        self.assertIn("TIME", tagged_words[3]["tags"])

    def test_caus(self):
        text = "they did also fall under the power of death , because they did eat in disobedience ; and disobedience to"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Because should be tagged as a CAUS
        self.assertIn("CAUS", tagged_words[10]["tags"])

    def test_conc(self):
        text = "outsider . When they learn you 're in the hills though , they 'll rally , do n't worry about"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Though should be tagged as a CONC
        self.assertIn("CONC", tagged_words[10]["tags"])

    def test_cond(self):
        text = "so high that the top falls gently over , as if to show that it really is hair and not"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # If should be tagged as a COND
        self.assertIn("COND", tagged_words[10]["tags"])

    def test_osub(self):
        text = (
            "his comment on the planter dynasties as they have existed since the decades before the Civil"
            " War . It may"
        )
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Since should be tagged as an OSUB
        self.assertIn("OSUB", tagged_words[10]["tags"])

    def test_osub_since(self):
        text = "his comment on the planter dynasties as they have existed since the decades before the Civil War . It may"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Since should be tagged as an OSUB
        self.assertIn("OSUB", tagged_words[10]["tags"])


if __name__ == "__main__":
    unittest.main()
