import unittest

import spacy

from biberplus.tagger import tag_text


class TestPassivesFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = spacy.load(
            "en_core_web_sm", disable=["parser", "lemmatizer", "ner"]
        )

    def test_pass(self):
        # Test basic BE + VBN pattern
        text1 = "The document was signed yesterday."
        tagged_words1 = tag_text(text1, pipeline=self.pipeline)
        self.assertIn("PASS", tagged_words1[2]["tags"])

        # Test BE + ADV + VBN pattern
        text2 = "The cake was quickly eaten."
        tagged_words2 = tag_text(text2, pipeline=self.pipeline)
        self.assertIn("PASS", tagged_words2[2]["tags"])

        # Test BE + ADV + ADV + VBN pattern
        text3 = "The message was very carefully written."
        tagged_words3 = tag_text(text3, pipeline=self.pipeline)
        self.assertIn("PASS", tagged_words3[2]["tags"])

        # Test BE + N/PRO + VBN pattern
        text4 = "The house is itself designed."
        tagged_words4 = tag_text(text4, pipeline=self.pipeline)
        self.assertIn("PASS", tagged_words4[2]["tags"])

    def test_bypa(self):
        # Test basic by-passive
        text1 = "The letter was written by John."
        tagged_words1 = tag_text(text1, pipeline=self.pipeline)
        self.assertIn("BYPA", tagged_words1[2]["tags"])

        # Test by-passive with intervening words
        text2 = "The book was carefully reviewed by the committee."
        tagged_words2 = tag_text(text2, pipeline=self.pipeline)
        self.assertIn("BYPA", tagged_words2[2]["tags"])

        # Test by-passive with multiple intervening words
        text3 = "The building was very thoroughly inspected by the experts."
        tagged_words3 = tag_text(text3, pipeline=self.pipeline)
        self.assertIn("BYPA", tagged_words3[2]["tags"])

        # Test by-passive with nominal form
        text4 = "The decision is itself influenced by recent events."
        tagged_words4 = tag_text(text4, pipeline=self.pipeline)
        self.assertIn("BYPA", tagged_words4[2]["tags"])


if __name__ == "__main__":
    unittest.main()
