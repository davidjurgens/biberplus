import unittest

import spacy

from biberplus.tagger import tag_text


class TestQuestionFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = spacy.load(
            "en_core_web_sm", disable=["parser", "lemmatizer", "ner"]
        )

    def test_whqu_why(self):
        text = "only ended up as one due to Columbia Records . Why did it bomb ? Because it 's awful ,"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Why should be tagged as WHQU
        self.assertIn("WHQU", tagged_words[10]["tags"])

    def test_whqu_what(self):
        text = "She was happy with her results . What are the consequences of that decision ? I'm not sure."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # What should be tagged as WHQU
        self.assertIn("WHQU", tagged_words[7]["tags"])

    def test_whqu_basic_forms(self):
        """Test basic WH-question forms with different WH-words"""
        texts = [
            ("The meeting ended early . What will happen next ?", "What"),
            ("She finished the project . When did you start ?", "When"),
            ("He seems confused . Where are you going ?", "Where"),
            ("The solution failed . How can we fix it ?", "How"),
            ("They made a decision . Which option did they choose ?", "Which"),
        ]

        for text, wh_word in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            word_index = next(
                i for i, w in enumerate(tagged_words) if w["text"] == wh_word
            )
            self.assertIn("WHQU", tagged_words[word_index]["tags"])

    def test_whqu_with_discourse_markers(self):
        """Test WH-questions with intervening discourse markers"""
        texts = [
            ("The system crashed . So , what should we do ?", "what"),
            ("It's complicated . Well , how does it work ?", "how"),
            ("That's interesting . Anyway , why did it fail ?", "why"),
            ("I'm confused . Now , where should we look ?", "where"),
        ]

        for text, wh_word in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            word_index = next(
                i for i, w in enumerate(tagged_words) if w["text"] == wh_word
            )
            self.assertIn("WHQU", tagged_words[word_index]["tags"])

    def test_whqu_negative_cases(self):
        """Test cases where WH-words should not be tagged as WHQU"""
        texts = [
            # Not questions but relative clauses
            "I know what they did.",
            "She understands how it works.",
            "They explained where to go.",
            # Not question-forming WH-words
            "However you look at it, it's wrong.",
            "Whatever happens, we'll be ready.",
            "Whenever you're ready, let's begin.",
        ]

        for text in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            wh_words = [
                w
                for w in tagged_words
                if w["text"].lower()
                in [
                    "what",
                    "where",
                    "when",
                    "how",
                    "which",
                    "why",
                    "whoever",
                    "whomever",
                    "whichever",
                    "wherever",
                    "whenever",
                    "whatever",
                    "however",
                ]
            ]
            for word in wh_words:
                self.assertNotIn("WHQU", word["tags"])

    def test_whqu_with_auxiliaries(self):
        """Test WH-questions with different auxiliary verbs"""
        texts = [
            ("The task is complex . What do we need?", "What"),  # DO auxiliary
            ("Things changed . What have they done?", "What"),  # HAVE auxiliary
            ("It's unclear . What will happen?", "What"),  # Modal auxiliary
            ("Nobody knows . What is going on?", "What"),  # BE auxiliary
        ]

        for text, wh_word in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            word_index = next(
                i for i, w in enumerate(tagged_words) if w["text"] == wh_word
            )
            self.assertIn("WHQU", tagged_words[word_index]["tags"])


if __name__ == "__main__":
    unittest.main()
