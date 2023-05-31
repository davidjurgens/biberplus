import unittest

import spacy

from bibermda.tagger import tag_text


class TestStativeFormFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_bema(self):
        text = 'have a little boy that likes tractors , you can be certain this will be entertaining to him ! '
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # 'Be' should be tagged as BEMA
        self.assertIn('BEMA', tagged_words[10]['tags'])

    def test_ex(self):
        pass


if __name__ == '__main__':
    unittest.main()
