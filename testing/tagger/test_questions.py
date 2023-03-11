import unittest

import spacy

from src.tagger.tagger_utils import build_variable_dictionaries
from src.tagger.word_tagger import WordTagger


class TestQuestionFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])
        self.patterns_dict = build_variable_dictionaries()

    def test_whqu(self):
        doc = self.pipeline("only ended up as one due to Columbia Records . Why did it bomb ? Because it 's awful ,")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Why should be tagged as WHQU
        self.assertIn('WHQU', tagger.tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
