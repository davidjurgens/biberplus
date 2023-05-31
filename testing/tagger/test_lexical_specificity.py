import unittest

import spacy

from bibermda.tagger.tagger_utils import build_variable_dictionaries
from bibermda.tagger.biber_tagger import BiberTagger


class TestLexicalSpecificityFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])
        self.patterns_dict = build_variable_dictionaries()

    def test_type_token_ratio(self):
        doc = self.pipeline("This is an example with repeated words. Words that occur more than once. "
                            "This is a simple example")

        word_tagger = BiberTagger(words=list(doc), patterns_dict=self.patterns_dict)
        word_tagger.run_all()
        self.assertEqual(word_tagger.ttr, 0.75)

    def test_mean_word_length(self):
        doc = self.pipeline("This is a simple test of words")
        word_tagger = BiberTagger(words=list(doc), patterns_dict=self.patterns_dict)
        word_tagger.run_all()
        self.assertEqual(round(word_tagger.mean_word_length, 2), 3.43)

    def test_word_count(self):
        doc = self.pipeline("I've read a few of these reviews and think that Fisher Price "
                            "must have a quality control issue .")
        word_tagger = BiberTagger(words=list(doc), patterns_dict=self.patterns_dict)
        word_tagger.run_all()
        self.assertEqual(word_tagger.word_count, 20)

    def test_total_adverbs(self):
        doc = self.pipeline("I quickly and intentionally came up with this example")
        word_tagger = BiberTagger(words=list(doc), patterns_dict=self.patterns_dict)
        word_tagger.run_all()
        self.assertEqual(word_tagger.adverb_count, 2)


if __name__ == '__main__':
    unittest.main()
