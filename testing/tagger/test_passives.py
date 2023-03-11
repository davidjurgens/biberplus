import unittest

import spacy

from src.tagger.tagger_utils import build_variable_dictionaries
from src.tagger.word_tagger import WordTagger


class TestPassivesFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])
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
        # onquered should be tagged as a BYPA
        self.assertIn('BYPA', tagger.tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
