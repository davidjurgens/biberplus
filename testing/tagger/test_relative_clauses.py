import unittest

import stanza

from src.tagger.tagger_utils import build_variable_dictionaries
from src.tagger.word_tagger import WordTagger


class TestRelativeClausesFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_tsub(self):
        doc = self.pipeline('we proceed through the seasons of life . Minor characters that surround the love '
                            'triangle are colorful and woven with')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # That should be tagged as a TSUB
        self.assertIn('TSUB', tagger.tagged_words[10]['tags'])

    def test_tobj(self):
        doc = self.pipeline('the dog that I saw')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # That should be tagged as a TOJB
        self.assertIn('TOBJ', tagger.tagged_words[2]['tags'])

    def test_whsub(self):
        doc = self.pipeline('. There are plenty of reference mentioned at the end which can be followed up '
                            'for more curiosity . Must')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Which should be tagged as a WHSUB
        self.assertIn('WHSUB', tagger.tagged_words[10]['tags'])

    def test_whobj(self):
        doc = self.pipeline("can be brave and courageous . Mafatu is a boy whose mom dies at sea and ever since he was")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Whose should be tagged as a WHOBJ
        self.assertIn('WHOBJ', tagger.tagged_words[10]['tags'])

    def test_pire(self):
        doc = self.pipeline('pencil ! I am a semi-professional singer , one of whose idols is the great '
                            'Judy Garland . No one')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Whose should be tagged as a PIRE
        self.assertIn('PIRE', tagger.tagged_words[11]['tags'])

    def test_sere(self):
        doc = self.pipeline('does not stop until you put the book down , which you will not do '
                            'until you have finished it')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Which should be tagged as a SERE
        self.assertIn('SERE', tagger.tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
