import unittest

import stanza

from src.tagger.tagger_utils import build_variable_dictionaries
from src.tagger.word_tagger import WordTagger


class TestAdjectiveFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_pred(self):
        doc = self.pipeline("and rambling . Yeah , these guys were profound and impressive "
                            "when I was in the 8th grade , but")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()

        # Impressive should be tagged as a PRED
        self.assertIn('PRED', tagger.tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
