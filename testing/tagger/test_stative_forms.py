import unittest

import stanza

from src.tagger.tagger_utils import build_variable_dictionaries
from src.tagger.word_tagger import WordTagger


class TestStativeFormFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_bema(self):
        doc = self.pipeline(
            'have a little boy that likes tractors , you can be certain this will be entertaining to him ! ')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # 'Be' should be tagged as BEMA
        self.assertIn('BEMA', tagger.tagged_words[10]['tags'])

    def test_ex(self):
        pass


if __name__ == '__main__':
    unittest.main()
