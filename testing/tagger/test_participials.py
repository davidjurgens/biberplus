import unittest

import stanza

from src.tagger.tagger_utils import build_variable_dictionaries
from src.tagger.word_tagger import WordTagger


class TesParticipialFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_presp(self):
        doc = self.pipeline('practice and for that it is a good resource . Knowing why some aspects '
                            'are not included and having the')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Knowing should be tagged as PRESP
        self.assertIn('PRESP', tagger.tagged_words[10]['tags'])

    def test_pastp(self):
        doc = self.pipeline('. Built in a single week, the house would stand for fifty years')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Built should be tagged as PASTP
        self.assertIn('PASTP', tagger.tagged_words[1]['tags'])

    def test_wzpast(self):
        doc = self.pipeline(
            'in most cases with understanding and restraint . The progress reported by the advisory committee is real . While some')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        for tagged in tagger.tagged_words:
            print(tagged)
        # Reported should be tagged as WZPAST
        self.assertIn('WZPAST', tagger.tagged_words[10]['tags'])

    def test_wzpres(self):
        doc = self.pipeline(
            "and the mean , and he sees the Compson family disintegrating from within . If the barn-burner 's family produces")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Disintegrating should be tagged as WZPAST
        self.assertIn('WZPREZ', tagger.tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
