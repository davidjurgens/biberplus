import unittest

import stanza

from src.tagger.tagger_utils import build_variable_dictionaries
from src.tagger.word_tagger import WordTagger


class TestNominalFormFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_nomz(self):
        doc = self.pipeline('consular materials to reveal the motives which led the British government to permit '
                            'Garibaldi to cross the Straits of Messina')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Government should be tagged as a NOMZ
        self.assertIn('NOMZ', tagger.tagged_words[10]['tags'])

    def test_ger(self):
        doc = self.pipeline('Democratic gubernatorial candidate , that the ~ GOP is `` Campaigning on the carcass of '
                            'Eisenhower Republicanism '' . Mitchell')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Campaigning should be tagged as a GER
        self.assertIn('GER', tagger.tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
