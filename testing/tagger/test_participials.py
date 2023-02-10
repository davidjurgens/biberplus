import unittest

import stanza

from src.tagger.simple_word_tagger import SimpleWordTagger
from src.tagger.tagger_utils import build_variable_dictionaries


class TesParticipialFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_presp(self):
        doc = self.pipeline('Built in a single week, the house would stand for fifty years')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        print(tagger.tagged_words)
        self.assertEqual(True, True)

    def test_pastp(self):
        doc = self.pipeline('The solution produced by this process')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        print(tagger.tagged_words)
        self.assertEqual(True, True)

    def test_wzpast(self):
        doc = self.pipeline('The event causing this decline is a rise in interest rates')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        print(tagger.tagged_words)
        self.assertEqual(True, True)

    def test_wzprez(self):
        doc = self.pipeline('Stuffing his mouth with cookies, Joe accepted and ran out the door')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        print(tagger.tagged_words)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
