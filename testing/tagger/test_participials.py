import unittest

import stanza

from src.tagger.simple_word_tagger import SimpleWordTagger
from src.tagger.tagger_utils import build_variable_dictionaries


class TesParticipialFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_presp(self):
        doc = self.pipeline(
            'of his early poems when he reads to jazz , including many of his Chinese and Japanese translations ; he')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Including should be tagged as PRESP
        self.assertIn('PRESP', tagger.tagged_words[10]['tags'])

    def test_pastp(self):
        doc = self.pipeline('rest her soul , she was a sweet one . Come on now '' . She put a strong hand')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Come should be tagged as PASTP
        self.assertIn('PRESP', tagger.tagged_words[10]['tags'])

    def test_wzpast(self):
        doc = self.pipeline(
            'in most cases with understanding and restraint . The progress reported by the advisory committee is real . While some')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Reported should be tagged as WZPAST
        self.assertIn('WZPAST', tagger.tagged_words[10]['tags'])

    def test_wzpres(self):
        doc = self.pipeline(
            "and the mean , and he sees the Compson family disintegrating from within . If the barn-burner 's family produces")
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Disintegrating should be tagged as WZPAST
        self.assertIn('WZPREZ', tagger.tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
