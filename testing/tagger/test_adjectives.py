import unittest

import stanza

from src.tagger.simple_word_tagger import SimpleWordTagger
from src.tagger.tagger_utils import build_variable_dictionaries


class TestAdjectiveFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_pred(self):
        doc = self.pipeline(
            'how to build those proofs . This film is reasonably good . '
            'The problem are two : Firstly , Clint')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Good should be tagged as a PRED
        self.assertIn('PRED', tagger.tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
