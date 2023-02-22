import unittest

import stanza

from src.tagger.word_tagger import WordTagger
from src.tagger.tagger_utils import build_variable_dictionaries


class TestDocTaggerFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_word_count(self):
        doc = self.pipeline("I 've read a few of these reviews and think that Fisher Price "
                            "must have a quality control issue .")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        self.assertEqual(len(tagger.words), 20)

    def test_mean_word_length(self):
        doc = self.pipeline("This is a simple test of words")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        self.assertEqual(round(tagger.mean_word_length, 2), 3.57)

    def test_total_adverbs(self):
        self.assertEqual(True, False)  # add assertion here

    def test_type_token_ratio(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
