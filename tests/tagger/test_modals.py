import unittest

import spacy

from biberplus.tagger import tag_text


class TestModalsFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = spacy.load(
            "en_core_web_sm", disable=["parser", "lemmatizer", "ner"]
        )

    def test_pomd(self):
        text = ", we 'll have to get organized . The baby can have an early nap . Victoria , I want"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Can should be tagged as a POMD
        self.assertIn("POMD", tagged_words[10]["tags"])

    def test_nemd(self):
        text = (
            "social values , it is clear that the educational profession must work for the values which "
            "are characteristic of the"
        )
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Must should be tagged as a NEMD
        self.assertIn("NEMD", tagged_words[10]["tags"])

    def test_prmd(self):
        text = "holds for values as well as life styles . One would need to test this proposition carefully ; after all"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Would should be tagged as a PRMD
        self.assertIn("PRMD", tagged_words[10]["tags"])

    def test_multiple_pomd(self):
        text = "You can go now. She may come later. We might need help. They could arrive soon."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Test each possibility modal
        self.assertIn("POMD", tagged_words[1]["tags"])  # can
        self.assertIn("POMD", tagged_words[6]["tags"])  # may
        self.assertIn("POMD", tagged_words[11]["tags"])  # might
        self.assertIn("POMD", tagged_words[16]["tags"])  # could

    def test_multiple_nemd(self):
        text = "You must leave now. She should study more. We ought to help."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Test each necessity modal
        self.assertIn("NEMD", tagged_words[1]["tags"])  # must
        self.assertIn("NEMD", tagged_words[6]["tags"])  # should
        self.assertIn("NEMD", tagged_words[11]["tags"])  # ought

    def test_multiple_prmd(self):
        text = "I will help. You would agree. He shall proceed. We'll continue. They'd come."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Test each predictive modal including contractions
        self.assertIn("PRMD", tagged_words[1]["tags"])  # will
        self.assertIn("PRMD", tagged_words[5]["tags"])  # would
        self.assertIn("PRMD", tagged_words[9]["tags"])  # shall
        self.assertIn("PRMD", tagged_words[13]["tags"])  # 'll
        self.assertIn("PRMD", tagged_words[17]["tags"])  # 'd

    def test_modal_negation(self):
        text = "You cannot go. She won't help. They shouldn't leave."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Modals should still be tagged when negated
        self.assertIn("POMD", tagged_words[1]["tags"])  # cannot

    def test_modal_questions(self):
        text = "Can you help? Would they agree? Must we leave?"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Modals should be tagged in questions
        self.assertIn("POMD", tagged_words[0]["tags"])  # Can
        self.assertIn("PRMD", tagged_words[4]["tags"])  # Would
        self.assertIn("NEMD", tagged_words[8]["tags"])  # Must


if __name__ == "__main__":
    unittest.main()
