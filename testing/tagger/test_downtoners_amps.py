import unittest

import stanza

from src.tagger.word_tagger import WordTagger
from src.tagger.tagger_utils import build_variable_dictionaries


class TestDownTonersAmpsFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_dwnt(self):
        doc = self.pipeline('a tangent point , and at such a point can only change by an even integer . '
                            'Thus the multiplicity')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Only should be tagged as a DWNT
        self.assertIn('DWNT', tagger.tagged_words[10]['tags'])

    def test_hdg(self):
        doc = self.pipeline('that blow to be borderline . To kayo him and maybe or maybe not kill . '
                            'You hit again about')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Maybe should be tagged as a HDG
        self.assertIn('HDG', tagger.tagged_words[10]['tags'])

    def test_amp(self):
        doc = self.pipeline('lie around on the rug during the meal , a very pretty sight as Rob Roy , ')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Very should be tagged as an AMP
        self.assertIn('AMP', tagger.tagged_words[10]['tags'])

    def test_emph(self):
        doc = self.pipeline('not be subjected to such a risk , or that such a possibility should '
                            'not be permitted to endanger the')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Such should be tagged as a EMPH
        self.assertIn('EMPH', tagger.tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
