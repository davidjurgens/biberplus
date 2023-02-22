import unittest

import stanza

from src.tagger.word_tagger import WordTagger
from src.tagger.tagger_utils import build_variable_dictionaries


class TestModalFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_pomd(self):
        doc = self.pipeline(", we 'll have to get organized . The baby can have an early nap . Victoria , I want")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Can should be tagged as a POMD
        self.assertIn('POMD', tagger.tagged_words[10]['tags'])

    def test_nemd(self):
        doc = self.pipeline('social values , it is clear that the educational profession must work for the'
                            ' values which are characteristic of the')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Must should be tagged as a NEMD
        self.assertIn('NEMD', tagger.tagged_words[10]['tags'])

    def test_prmd(self):
        doc = self.pipeline('holds for values as well as life styles . One would need to test '
                            'this proposition carefully ; after all')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Would should be tagged as a PRMD
        self.assertIn('PRMD', tagger.tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
