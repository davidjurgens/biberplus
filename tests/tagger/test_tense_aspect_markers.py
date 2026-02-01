import unittest

import spacy

from biberplus.tagger import tag_text


class TestTenseAspectFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = spacy.load(
            "en_core_web_sm", disable=["parser", "lemmatizer", "ner"]
        )

    def test_peas(self):
        text = (
            "the exchange , so far all my dealings with amazon have been easy and fair . "
            "Waterloo Station is a"
        )
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Have should be tagged as PEAS
        self.assertIn("PEAS", tagged_words[10]["tags"])

    def test_peas_direct_verb(self):
        text = "They have visited the museum several times."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Have should be tagged as PEAS
        self.assertIn("PEAS", tagged_words[1]["tags"])

    def test_peas_one_intervening_word(self):
        text = "I have always loved that painting."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Have should be tagged as PEAS
        self.assertIn("PEAS", tagged_words[1]["tags"])

    def test_peas_two_intervening_adverbs(self):
        text = "She has probably never been to the opera."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Has should be tagged as PEAS
        self.assertIn("PEAS", tagged_words[1]["tags"])

    def test_past_tense_simple(self):
        """Test simple past tense verbs"""
        texts = [
            "She walked to the store",
            "They ran the experiment",
            "The results showed significant differences",
        ]
        for text in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            past_verbs = [
                word
                for word in tagged_words
                if word["text"] in ["walked", "ran", "showed"]
            ]
            for verb in past_verbs:
                self.assertIn("VBD", verb["tags"])

    def test_perfect_aspect_basic(self):
        """Test basic perfect aspect constructions"""
        texts = [
            ("I have finished the report", 1),
            ("They have completed the study", 1),
            ("The researchers have published their findings", 2),
        ]
        for text, have_index in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            # Have should be tagged as PEAS
            self.assertIn("PEAS", tagged_words[have_index]["tags"])

    def test_perfect_aspect_with_adverbs(self):
        """Test perfect aspect with intervening adverbs"""
        texts = [
            "We have successfully implemented the system",
            "They have carefully analyzed the data",
            "She has thoroughly reviewed the literature",
        ]
        for text in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            self.assertIn("PEAS", tagged_words[1]["tags"])  # have

    def test_perfect_aspect_with_negation(self):
        """Test perfect aspect with negation"""
        texts = [
            ("I have not seen the results", 1),
            ("They haven't completed the analysis", 1),  # haven't is one token
            ("The team has not published their findings", 2),
        ]
        for text, have_index in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            self.assertIn("PEAS", tagged_words[have_index]["tags"])  # have/has

    def test_present_tense_third_person(self):
        """Test present tense verbs (third person)"""
        texts = [
            "She writes clearly",
            "The system works well",
            "This paper describes the method",
        ]
        for text in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            present_verbs = [
                word
                for word in tagged_words
                if word["text"] in ["writes", "works", "describes"]
            ]
            for verb in present_verbs:
                self.assertIn("VPRT", verb["tags"])

    def test_present_tense_non_third_person(self):
        """Test present tense verbs (non-third person)"""
        texts = ["I write papers", "We analyze data", "They study behavior"]
        for text in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            present_verbs = [
                word
                for word in tagged_words
                if word["text"] in ["write", "analyze", "study"]
            ]
            for verb in present_verbs:
                self.assertIn("VPRT", verb["tags"])

    def test_complex_tense_combinations(self):
        """Test sentences with multiple tense/aspect markers"""
        text = "While I have analyzed the data, she writes the conclusion, and they completed the introduction."
        tagged_words = tag_text(text, pipeline=self.pipeline)

        # Check perfect aspect
        self.assertIn("PEAS", tagged_words[2]["tags"])  # have

        # Check present tense
        self.assertIn("VPRT", tagged_words[8]["tags"])  # writes


if __name__ == "__main__":
    unittest.main()
