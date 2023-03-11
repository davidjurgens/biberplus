import unittest

import spacy

from src.tagger.word_tagger import WordTagger
from src.tagger.tagger_utils import build_variable_dictionaries


class TestNegationFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])
        self.patterns_dict = build_variable_dictionaries()

    def test_syne(self):
        doc = self.pipeline("I did n't even want to give it away . No Beatrix Potter appreciator should "
                            "be exposed to this ;")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        self.assertIn('SYNE', tagger.tagged_words[10]['tags'])

    def test_xx0(self):
        doc = self.pipeline(', and General Motors on the other . It is not a medieval mental quirk '
                            'or an attitude `` unnourished')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Not should be tagged as a XXO
        self.assertIn('XX0', tagger.tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
