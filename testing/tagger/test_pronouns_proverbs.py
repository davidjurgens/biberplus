import unittest

import spacy

from src.tagger.tagger_main import run_tagger_on_string


class TestPronounProverbFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_fpp1(self):
        text = 'the soil soft during these early days of growth . I like sawdust for this , or hay . When'
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # I should be tagged as a FPP1
        self.assertIn('FPP1', tagged_words[10]['tags'])

    def test_spp2(self):
        text = ". . -RRB- The new interpretation makes sense though if you think about it . " \
               "By the way , he"
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # You should be tagged as a SPP2
        self.assertIn('SPP2', tagged_words[10]['tags'])

    def test_tpp3(self):
        text = 'a child till he was sixteen , a youth till he was five-and-twenty , and a young man till he'
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # He should be tagged as a TPP3
        self.assertIn('TPP3', tagged_words[10]['tags'])

    def test_pit(self):
        text = 'sometimes answers itself , and that the way in which it is posed frequently shapes ' \
               'the answer . Chewing it'
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # It should be tagged as a PIT
        self.assertIn('PIT', tagged_words[10]['tags'])

    def test_demp(self):
        text = 'Vernon on the morning of the regular tallyho run . This was an honor , ' \
               'like dining with a captain'
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # This should be tagged as a DEMP
        self.assertIn('DEMP', tagged_words[10]['tags'])

    def test_inpr(self):
        text = "I turned away from her coldly . `` It was nobody 's fault . She overplayed her hand '' ."
        tagged_words = run_tagger_on_string(self.pipeline, text)
        # Nobody should be tagged as a INPR
        self.assertIn('INPR', tagged_words[11]['tags'])

    def test_prod(self):
        pass


if __name__ == '__main__':
    unittest.main()
