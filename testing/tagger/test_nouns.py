import unittest

import stanza

from src.tagger.simple_word_tagger import SimpleWordTagger
from src.tagger.tagger_utils import build_variable_dictionaries


class TestNounFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_fpp1(self):
        doc = self.pipeline('the soil soft during these early days of growth . I like sawdust for this , or hay . When')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # I should be tagged as a FPP1
        self.assertIn('FPP1', tagger.tagged_words[10]['tags'])

    def test_spp2(self):
        doc = self.pipeline(
            'one of the most extraordinary views in Rome . If you look through the keyhole , you will see an')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # You should be tagged as a SPP2
        self.assertIn('SPP2', tagger.tagged_words[10]['tags'])

    def test_tpp3(self):
        doc = self.pipeline(
            'a child till he was sixteen , a youth till he was five-and-twenty , and a young man till he')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # He should be tagged as a TPP3
        self.assertIn('TPP3', tagger.tagged_words[10]['tags'])

    def test_pit(self):
        doc = self.pipeline(
            'sometimes answers itself , and that the way in which it is posed frequently shapes the answer . Chewing it')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # It should be tagged as a PIT
        self.assertIn('PRIV', tagger.tagged_words[10]['tags'])

    def test_demp(self):
        doc = self.pipeline('Vernon on the morning of the regular tallyho run . This was an honor , '
                            'like dining with a captain')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # This should be tagged as a DEMP
        self.assertIn('DEMP', tagger.tagged_words[10]['tags'])

    def test_inpr(self):
        doc = self.pipeline("I turned away from her coldly . `` It was nobody 's fault . She overplayed her hand '' .")
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Nobody should be tagged as a INPR
        self.assertIn('INPR', tagger.tagged_words[10]['tags'])

    def test_nomz(self):
        doc = self.pipeline('consular materials to reveal the motives which led the British government to permit '
                            'Garibaldi to cross the Straits of Messina')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Government should be tagged as a NOMZ
        self.assertIn('NOMZ', tagger.tagged_words[10]['tags'])

    def test_ger(self):
        doc = self.pipeline('Democratic gubernatorial candidate , that the ~ GOP is `` Campaigning on the carcass of '
                            'Eisenhower Republicanism '' . Mitchell')
        tagger = SimpleWordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Campaigning should be tagged as a GER
        self.assertIn('GER', tagger.tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
