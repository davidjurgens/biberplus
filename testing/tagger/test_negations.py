import unittest

import stanza

from src.tagger.simple_word_tagger import SimpleWordTagger
from src.tagger.tagger_utils import build_variable_dictionaries


class TestNegationFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_syne(self):
        doc = self.pipeline('small fortune on a search for him but had made no provision '
                            'for him in her will if he should')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # No should be tagged as a SYNE
        self.assertIn('SYNE', tagger.tagged_words[10]['tags'])

    def test_xxo(self):
        doc = self.pipeline(', and General Motors on the other . It is not a medieval mental quirk '
                            'or an attitude `` unnourished')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Not should be tagged as a XXO
        self.assertIn('XXO', tagger.tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
