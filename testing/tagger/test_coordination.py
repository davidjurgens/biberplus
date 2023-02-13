import unittest

import stanza

from src.tagger.simple_word_tagger import SimpleWordTagger
from src.tagger.tagger_utils import build_variable_dictionaries


class TestCoordinationFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_phc(self):
        doc = self.pipeline("I dont think it was as good as Suicide Notes and Butterfly Kisses . "
                            "They should try to be as")
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # And should be tagged as a PHC
        self.assertIn('PHC', tagger.tagged_words[10]['tags'])

    def test_andc(self):
        doc = self.pipeline("to write a book that has already been written , and she fails in comparison . "
                            "Spend your money on")
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # And should be tagged as a ANDC
        self.assertIn('ANDC', tagger.tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
