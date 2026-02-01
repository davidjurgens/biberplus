import unittest

import spacy

from biberplus.tagger import tag_text


class TestPrepPhrasesAdjectiveAdverbFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = spacy.load(
            "en_core_web_sm", disable=["parser", "lemmatizer", "ner"]
        )

    def test_pin(self):
        text = "have kept my hair in great condition ! A waste of money that 's all I have to say about"
        tagged_words = tag_text(text, pipeline=self.pipeline)

        # Of should be tagged as PIN
        self.assertIn("PIN", tagged_words[10]["tags"])

    def test_pred(self):
        text = "and rambling . Yeah , these guys were profound and impressive when I was in the 8th grade , but"
        tagged_words = tag_text(text, pipeline=self.pipeline)

        # Impressive should be tagged as a PRED
        self.assertIn("PRED", tagged_words[10]["tags"])

    def test_pred_impressive(self):
        text = "and rambling . Yeah , these guys were profound and impressive when I was in the 8th grade , but"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Impressive should be tagged as a PRED
        self.assertIn("PRED", tagged_words[10]["tags"])

    def test_pred_magnificent(self):
        text = "The flowers are beautiful but short-lived. The building is tall and magnificent."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Beautiful and magnificent should be tagged as a PRED
        self.assertIn("PRED", tagged_words[3]["tags"])
        self.assertIn("PRED", tagged_words[-2]["tags"])

    def test_pin_multiple(self):
        text = "The book on the shelf in the corner of the room was dusty"
        tagged_words = tag_text(text, pipeline=self.pipeline)

        # Check multiple prepositions are tagged as PIN
        self.assertIn("PIN", tagged_words[2]["tags"])  # on
        self.assertIn("PIN", tagged_words[5]["tags"])  # in
        self.assertIn("PIN", tagged_words[8]["tags"])  # of

    def test_pred_be_forms(self):
        text = "The flowers are beautiful. The cat was lazy. The students have been diligent."
        tagged_words = tag_text(text, pipeline=self.pipeline)

        # Test predicative adjectives with different BE forms
        self.assertIn("PRED", tagged_words[3]["tags"])  # beautiful
        self.assertIn("PRED", tagged_words[8]["tags"])  # lazy
        self.assertIn("PRED", tagged_words[14]["tags"])  # diligent

    def test_rb_adverbs(self):
        text = "She quickly and efficiently completed the task. He spoke very softly."
        tagged_words = tag_text(text, pipeline=self.pipeline)

        # Test different types of adverbs
        self.assertIn("RB", tagged_words[1]["tags"])  # quickly
        self.assertIn("RB", tagged_words[3]["tags"])  # efficiently
        self.assertIn("RB", tagged_words[10]["tags"])  # very
        self.assertIn("RB", tagged_words[11]["tags"])  # softly


if __name__ == "__main__":
    unittest.main()
