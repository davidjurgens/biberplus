import unittest

import spacy

from biberplus.tagger import tag_text


class TestPrepPhrasesAdjectiveAdverbFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_pin(self):
        text = "have kept my hair in great condition ! A waste of money that 's all I have to say about"
        tagged_words = tag_text(text, pipeline=self.pipeline)

        # Of should be tagged as PIN
        self.assertIn('PIN', tagged_words[10]['tags'])

    def test_pred(self):
        text = "and rambling . Yeah , these guys were profound and impressive when I was in the 8th grade , but"
        tagged_words = tag_text(text, pipeline=self.pipeline)

        # Impressive should be tagged as a PRED
        self.assertIn('PRED', tagged_words[10]['tags'])

    def test_pred_impressive(self):
        text = "and rambling . Yeah , these guys were profound and impressive when I was in the 8th grade , but"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Impressive should be tagged as a PRED
        self.assertIn('PRED', tagged_words[10]['tags'])

    def test_pred_magnificent(self):
        text = "The flowers are beautiful but short-lived. The building is tall and magnificent."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Beautiful and magnificent should be tagged as a PRED
        self.assertIn('PRED', tagged_words[3]['tags'])
        self.assertIn('PRED', tagged_words[-2]['tags'])


if __name__ == '__main__':
    unittest.main()
