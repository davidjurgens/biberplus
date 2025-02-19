import unittest

import spacy

from biberplus.tagger import tag_text


class TestNominalFormFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_nomz_basic(self):
        text = 'consular materials to reveal the motives which led the British government to permit Garibaldi to cross ' \
               'the Straits of Messina'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Government should be tagged as a NOMZ
        self.assertIn('NOMZ', tagged_words[10]['tags'])

    def test_ger(self):
        text = "His voice carries the album well even with some subpar songwriting . I do n't know where " \
               "people are getting"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Songwriting should be tagged as a GER
        self.assertIn('GER', tagged_words[10]['tags'])

    def test_nomz_tion(self):
        text = "The administration's decision led to much frustration in the population"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Test multiple -tion nominalizations
        self.assertIn('NOMZ', tagged_words[1]['tags'])  # administration
        self.assertIn('NOMZ', tagged_words[3]['tags'])  # decision
        self.assertIn('NOMZ', tagged_words[7]['tags'])  # frustration
        self.assertIn('NOMZ', tagged_words[10]['tags'])  # population

    def test_nomz_ment(self):
        text = "The government's assessment of the development and management situation"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Test -ment nominalizations
        self.assertIn('NOMZ', tagged_words[3]['tags'])  # assessment
        self.assertIn('NOMZ', tagged_words[6]['tags'])  # development
        self.assertIn('NOMZ', tagged_words[8]['tags'])  # management

    def test_nomz_ness(self):
        text = "The darkness and loneliness led to great sadness"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Test -ness nominalizations
        self.assertIn('NOMZ', tagged_words[1]['tags'])  # darkness
        self.assertIn('NOMZ', tagged_words[3]['tags'])  # loneliness
        self.assertIn('NOMZ', tagged_words[7]['tags'])  # sadness

    def test_nomz_ity(self):
        text = "The possibility of activity depends on ability and creativity"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Test -ity nominalizations
        self.assertIn('NOMZ', tagged_words[1]['tags'])  # possibility
        self.assertIn('NOMZ', tagged_words[3]['tags'])  # activity
        self.assertIn('NOMZ', tagged_words[6]['tags'])  # ability
        self.assertIn('NOMZ', tagged_words[8]['tags'])  # creativity

    def test_ger_complex(self):
        text = "Understanding and implementing these changes requires careful planning"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Test multiple gerunds (words > 10 chars)
        self.assertIn('GER', tagged_words[0]['tags'])  # understanding
        self.assertIn('GER', tagged_words[2]['tags'])  # implementing

    def test_ger_short_words(self):
        text = "The running and jumping exercises were too demanding"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Short words (< 10 chars) should not be tagged as gerunds
        self.assertNotIn('GER', tagged_words[1]['tags'])  # running
        self.assertNotIn('GER', tagged_words[3]['tags'])  # jumping

if __name__ == '__main__':
    unittest.main()
