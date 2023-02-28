import unittest

import stanza

from src.tagger.word_tagger import WordTagger
from src.tagger.tagger_utils import build_variable_dictionaries


class TestComplementsFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_thvc(self):
        doc = self.pipeline("I 've read a few of these reviews and think that Fisher Price "
                            "must have a quality control issue .")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # That should be tagged as a THVC
        self.assertIn('THVC', tagger.tagged_words[10]['tags'])

    def test_thac(self):
        doc = self.pipeline("twice a day for 20 minutes per use . Disappointing that it failed so quickly "
                            ". I have now owned")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # That should be tagged as a THAC
        self.assertIn('THAC', tagger.tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
