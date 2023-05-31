import unittest

import spacy

from bibermda.tagger import tag_text


class TestQuestionFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_whqu(self):
        text = "only ended up as one due to Columbia Records . Why did it bomb ? Because it 's awful ,"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Why should be tagged as WHQU
        self.assertIn('WHQU', tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
