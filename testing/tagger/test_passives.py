import unittest

import stanza

from src.tagger.word_tagger import WordTagger
from src.tagger.tagger_utils import build_variable_dictionaries


class TestPassiveFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_pass(self):
        doc = self.pipeline('sound scape is great , I am hearing nuance that was never heard before - '
                            'an otherwise perfect headphone for')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Was should be tagged as a PASS
        self.assertIn('PASS', tagger.tagged_words[10]['tags'])

    def test_bypa(self):
        doc = self.pipeline('well after the Egyptian Golden Years , after Egypt had been conquered by '
                            'the Greeks ... a time during which')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # By should be tagged as a BYPA
        self.assertIn('BYPA', tagger.tagged_words[12]['tags'])


if __name__ == '__main__':
    unittest.main()
