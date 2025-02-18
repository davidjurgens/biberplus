import unittest

import spacy

from biberplus.tagger import tag_text


class TestSpecializedVerbFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_priv(self):
        text = 'I expected to see ten, but instead saw twelve'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Expected should be tagged as a priv
        self.assertIn('PRIV', tagged_words[1]['tags'])

    def test_pubv(self):
        text = 'The candidate conceded the election late last night'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Conceded should be tagged as a pubv
        self.assertIn('PUBV', tagged_words[2]['tags'])

    def test_suav(self):
        text = 'I proposed extending the deadline'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Proposed should be tagged as a suav
        self.assertIn('SUAV', tagged_words[1]['tags'])

    def test_smp(self):
        text = 'edge of the bank . From the outside , it seemed no more than a low drumlin , a lump'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Seemed should be tagged as SMP
        self.assertIn('SMP', tagged_words[10]['tags'])

    def test_priv_multiple_forms(self):
        """Test different forms of private verbs"""
        test_cases = [
            ("I believe in science", 1),
            ("She thinks about the future", 1),
            ("They assumed it would work", 1),
            ("We are considering the proposal", 2),
            ("He felt the temperature rising", 1)
        ]
        for text, verb_index in test_cases:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            # Check that the main verb is tagged as PRIV
            self.assertIn('PRIV', tagged_words[verb_index]['tags'])

    def test_pubv_multiple_forms(self):
        """Test different forms of public verbs"""
        test_cases = [
            ("The spokesperson announced the results", 2),
            ("They claimed victory in the election", 1),
            ("She replied to the accusations", 1),
            ("The witness testified in court", 2),
            ("The document states clearly that", 2)
        ]
        for text, verb_index in test_cases:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            # Check that the main verb is tagged as PUBV
            self.assertIn('PUBV', tagged_words[verb_index]['tags'])

    def test_suav_multiple_forms(self):
        """Test different forms of suasive verbs"""
        test_cases = [
            ("The committee recommended changes", 2),
            ("We demand immediate action", 1),
            ("They urged caution", 1),
            ("The manager suggested improvements", 2),
            ("The policy requires compliance", 2)
        ]
        for text, verb_index in test_cases:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            # Check that the main verb is tagged as SUAV
            self.assertIn('SUAV', tagged_words[verb_index]['tags'])

    def test_smp_multiple_forms(self):
        """Test different forms of seem/appear"""
        test_cases = [
            ("It seems unlikely", 1),
            ("The solution appeared effective", 2),
            ("Everything seems to work", 1),
            ("The results appear conclusive", 2),
            ("That seemed strange to me", 1)
        ]
        for text, verb_index in test_cases:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            # Check that seem/appear is tagged as SMP
            self.assertIn('SMP', tagged_words[verb_index]['tags'])

    def test_priv_with_negation(self):
        """Test private verbs with negation"""
        text = "I don't believe their story"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('PRIV', tagged_words[3]['tags'])

    def test_pubv_with_that_complement(self):
        """Test public verbs with that-complement"""
        text = "The report stated that the findings were significant"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('PUBV', tagged_words[2]['tags'])

    def test_suav_with_to_infinitive(self):
        """Test suasive verbs with to-infinitive"""
        text = "The board proposed to increase funding"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('SUAV', tagged_words[2]['tags'])

    def test_smp_with_to_complement(self):
        """Test seem/appear with to-complement"""
        text = "The project appears to be successful"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('SMP', tagged_words[2]['tags'])

    def test_multiple_specialized_verbs(self):
        """Test text with multiple specialized verbs"""
        text = "I think they claimed that we should propose new solutions, which seems reasonable"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('PRIV', tagged_words[1]['tags'])  # think
        self.assertIn('PUBV', tagged_words[3]['tags'])  # claimed
        self.assertIn('SUAV', tagged_words[7]['tags'])  # propose
        self.assertIn('SMP', tagged_words[12]['tags'])  # seems


if __name__ == '__main__':
    unittest.main()
