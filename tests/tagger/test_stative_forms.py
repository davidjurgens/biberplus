import unittest

import spacy

from biberplus.tagger import tag_text


class TestStativeFormFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = spacy.load(
            "en_core_web_sm", disable=["parser", "lemmatizer", "ner"]
        )

    def test_bema(self):
        text = "have a little boy that likes tractors , you can be certain this will be entertaining to him ! "
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # 'Be' should be tagged as BEMA
        self.assertIn("BEMA", tagged_words[10]["tags"])

    def test_bema_directly_followed(self):
        text = "They are wonderful people to work with."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # 'Are' should be tagged as BEMA because it's followed directly by an adjective (JJ) - "wonderful".
        self.assertIn("BEMA", tagged_words[1]["tags"])

    def test_bema_with_determiners(self):
        """Test BE as main verb followed by determiners"""
        texts = ["This is the answer", "Those are my books", "That was their decision"]
        for text in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            self.assertIn("BEMA", tagged_words[1]["tags"])

    def test_bema_with_adjectives(self):
        """Test BE as main verb followed by adjectives"""
        text = "The results are significant. The test was successful. They were happy."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn("BEMA", tagged_words[2]["tags"])  # are
        self.assertIn("BEMA", tagged_words[7]["tags"])  # was
        self.assertIn("BEMA", tagged_words[11]["tags"])  # were

    def test_bema_with_prepositions(self):
        """Test BE as main verb followed by prepositions"""
        text = "The book is on the shelf. They were in the garden."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn("BEMA", tagged_words[2]["tags"])  # is
        self.assertIn("BEMA", tagged_words[8]["tags"])  # were

    def test_pass_simple(self):
        """Test agentless passive constructions"""
        texts = [
            "The paper was published last year",
            "The data were collected carefully",
            "The building is being renovated",
        ]
        for text in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            self.assertIn("PASS", tagged_words[2]["tags"])  # was/were/is

    def test_pass_with_adverbs(self):
        """Test passives with intervening adverbs"""
        text = "The results were carefully analyzed. The text is quickly processed."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn("PASS", tagged_words[2]["tags"])  # were
        self.assertIn("PASS", tagged_words[8]["tags"])  # is

    def test_bypa_identification(self):
        """Test by-passive constructions"""
        text = "The book was written by the author. The study was conducted by researchers."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn("BYPA", tagged_words[2]["tags"])  # was
        self.assertIn("BYPA", tagged_words[10]["tags"])  # was

    def test_ex_there(self):
        """Test existential there constructions"""
        texts = [
            "There is a problem",
            "There are many solutions",
            "There were several issues",
        ]
        for text in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            self.assertIn("EX", tagged_words[0]["tags"])  # there
            self.assertNotIn(
                "BEMA", tagged_words[1]["tags"]
            )  # is/are/were should not be tagged as BEMA

    def test_bema_negation(self):
        """Test BE as main verb with negation"""
        text = "The solution is not obvious. They are not in the office."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn("BEMA", tagged_words[2]["tags"])  # is
        self.assertIn("BEMA", tagged_words[7]["tags"])  # are


if __name__ == "__main__":
    unittest.main()
