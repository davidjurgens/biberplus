import unittest

import spacy

from biberplus.tagger import tag_text


class TestNegationFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_syne(self):
        text = "I did n't even want to give it away . No Beatrix Potter appreciator should " \
               "be exposed to this ;"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('SYNE', tagged_words[10]['tags'])

    def test_syne_neith(self):
        text = "I have neither skill nor patience for this task."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Neither should be tagged as SYNE before skill
        self.assertIn('SYNE', tagged_words[2]['tags'])

    def test_syne_no(self):
        text = "No good deed goes unpunished."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # No should be tagged as SYNE before good
        self.assertIn('SYNE', tagged_words[0]['tags'])

    def test_xx0(self):
        text = ', and General Motors on the other . It is not a medieval mental quirk or an attitude `` unnourished'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Not should be tagged as a XXO
        self.assertIn('XX0', tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
