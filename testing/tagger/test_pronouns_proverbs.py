import unittest

import spacy

from src.tagger.tagger_utils import build_variable_dictionaries
from src.tagger.word_tagger import WordTagger


class TestPronounProverbFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])
        self.patterns_dict = build_variable_dictionaries()

    def test_fpp1(self):
        doc = self.pipeline('the soil soft during these early days of growth . I like sawdust for this , or hay . When')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # I should be tagged as a FPP1
        self.assertIn('FPP1', tagger.tagged_words[10]['tags'])

    def test_spp2(self):
        doc = self.pipeline(". . -RRB- The new interpretation makes sense though if you think about it . "
                            "By the way , he")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # You should be tagged as a SPP2
        self.assertIn('SPP2', tagger.tagged_words[10]['tags'])

    def test_tpp3(self):
        doc = self.pipeline(
            'a child till he was sixteen , a youth till he was five-and-twenty , and a young man till he')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # He should be tagged as a TPP3
        self.assertIn('TPP3', tagger.tagged_words[10]['tags'])

    def test_pit(self):
        doc = self.pipeline(
            'sometimes answers itself , and that the way in which it is posed frequently shapes the answer . Chewing it')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # It should be tagged as a PIT
        self.assertIn('PIT', tagger.tagged_words[10]['tags'])

    def test_demp(self):
        doc = self.pipeline('Vernon on the morning of the regular tallyho run . This was an honor , '
                            'like dining with a captain')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # This should be tagged as a DEMP
        self.assertIn('DEMP', tagger.tagged_words[10]['tags'])

    def test_inpr(self):
        doc = self.pipeline("I turned away from her coldly . `` It was nobody 's fault . She overplayed her hand '' .")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Nobody should be tagged as a INPR
        self.assertIn('INPR', tagger.tagged_words[11]['tags'])

    def test_prod(self):
        pass


if __name__ == '__main__':
    unittest.main()
