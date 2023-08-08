import unittest

import spacy

from biberplus.tagger import tag_text


class TestQuestionFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_whqu_why(self):
        text = "only ended up as one due to Columbia Records . Why did it bomb ? Because it 's awful ,"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Why should be tagged as WHQU
        self.assertIn('WHQU', tagged_words[10]['tags'])

    def test_whqu_what(self):
        text = "She was happy with her results . What are the consequences of that decision ? I'm not sure."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # What should be tagged as WHQU
        self.assertIn('WHQU', tagged_words[7]['tags'])


if __name__ == '__main__':
    unittest.main()
