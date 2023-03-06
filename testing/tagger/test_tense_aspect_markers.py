import unittest

import stanza

from src.tagger.tagger_utils import build_variable_dictionaries
from src.tagger.word_tagger import WordTagger


class TestTenseAspectFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_peas(self):
        doc = self.pipeline("the exchange , so far all my dealings with amazon have been easy and fair . "
                            "Waterloo Station is a")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Have should be tagged as PEAS
        self.assertIn('PEAS', tagger.tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
