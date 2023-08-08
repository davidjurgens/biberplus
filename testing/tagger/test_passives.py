import unittest

import spacy

from bibermda.tagger import tag_text


class TestPassivesFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_pass(self):
        text = 'sound scape is great , I am hearing nuance that was never heard before - an otherwise ' \
               'perfect headphone for'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Was should be tagged as a PASS
        self.assertIn('PASS', tagged_words[10]['tags'])

        # Test 1
        text1 = 'sound scape is great, I am hearing nuance that was never heard before - an otherwise ' \
                'perfect headphone for'
        tagged_words1 = tag_text(text1, pipeline=self.pipeline)
        # 'Was' should be tagged as a PASS
        self.assertIn('PASS', tagged_words1[10]['tags'])

        # Test 2
        text2 = 'The cake was quickly eaten by the children during the party, and the room was left in a mess.'
        tagged_words2 = tag_text(text2, pipeline=self.pipeline)
        # 'Was' should be tagged as a PASS
        self.assertIn('PASS', tagged_words2[2]['tags'])

    def test_bypa(self):
        text = 'well after the Egyptian Golden Years , after Egypt had been conquered by the Greeks ... ' \
               'a time during which'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # onquered should be tagged as a BYPA
        self.assertIn('BYPA', tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
