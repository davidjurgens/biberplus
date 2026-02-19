import unittest

import spacy

from biberplus.tagger import tag_text


class TestAdditionalFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_articles(self):
        text = "The cat chased a mouse across the room."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('ART', tagged_words[0]['tags'])
        self.assertIn('ART', tagged_words[3]['tags'])
        self.assertIn('ART', tagged_words[6]['tags'])

    def test_auxillary_be(self):
        text = "You have been standing"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('AUXB', tagged_words[2]['tags'])

    def test_capitalizations(self):
        text = "The Eiffel Tower in Paris is a famous landmark."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('CAP', tagged_words[1]['tags'])
        self.assertIn('CAP', tagged_words[2]['tags'])
        self.assertIn('CAP', tagged_words[4]['tags'])

    def test_subordinating_conjunctions(self):
        text = "When the doorbell rang, my dog Skeeter barked loudly."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('SCONJ', tagged_words[0]['tags'])

    def test_coordinating_conjunctions(self):
        text = "I wanted to go swimming, but it started raining."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('CCONJ', tagged_words[6]['tags'])

    def test_determiners(self):
        text = "Please pass me the book on the table."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('DET', tagged_words[3]['tags'])
        self.assertIn('DET', tagged_words[6]['tags'])

    def test_emoji(self):
        text = "I'm so excited for the party tonight! 🎉"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('EMOJ', tagged_words[-1]['tags'])

    def test_emoticon(self):
        text = "She received good news today! :)"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('EMOT', tagged_words[-1]['tags'])

    def test_exclamation_mark(self):
        text = "What a beautiful sunset!"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('EXCL', tagged_words[-1]['tags'])

    def test_hashtag(self):
        text = "I love going hiking in the #mountains."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('HASH', tagged_words[6]['tags'])

    def test_infinitives(self):
        text = "She plans to travel around the world."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('INF', tagged_words[3]['tags'])

    def test_interjections(self):
        text = "Wow, that was an amazing performance!"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('UH', tagged_words[0]['tags'])

    def test_numerals(self):
        text = "I have three dogs and two cats."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('NUM', tagged_words[2]['tags'])
        self.assertIn('NUM', tagged_words[5]['tags'])

    def test_laughter_acronyms(self):
        text = "His joke was so funny, LOL!"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('LAUGH', tagged_words[-2]['tags'])

    def test_interjections_middle(self):
        text = "I mean gosh, are you really sure about that?"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        print(tagged_words)
        self.assertIn('UH', tagged_words[2]['tags'])

    def test_interjections_end(self):
        text = "That's a bad idea, gosh"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('UH', tagged_words[-1]['tags'])

    def test_possessive_pronoun(self):
        text = "That car is hers, not mine."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('PRP', tagged_words[3]['tags'])

    def test_preposition(self):
        text = "He walked through the park to get to the other side."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('PREP', tagged_words[2]['tags'])

    def test_proper_noun(self):
        text = "John went to visit the Statue of Liberty in New York City."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('NNP', tagged_words[0]['tags'])

    def test_question_mark(self):
        text = "This sentence ends in a question mark?"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('QUES', tagged_words[-1]['tags'])

    def test_quotation_mark(self):
        text = '"Life is either a daring adventure or nothing at all." - Helen Keller'
        tagged_words = tag_text(text, pipeline=self.pipeline)

        self.assertIn('QUOT', tagged_words[0]['tags'])
        self.assertIn('QUOT', tagged_words[12]['tags'])

    def test_at(self):
        text = "Please email me your report @johnsmith."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('AT', tagged_words[-2]['tags'])

    def test_subject_pronouns(self):
        text = "She is going to the store."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('SBJP', tagged_words[0]['tags'])

    def test_subject_pronouns_end(self):
        text = "The person going to the store is she"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('SBJP', tagged_words[-1]['tags'])

    def test_url(self):
        text = "Check out the latest news on our website: www.example.com."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('URL', tagged_words[-2]['tags'])

    def test_url_org(self):
        text = "Check out the latest news on our website: https://website.org"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('URL', tagged_words[-1]['tags'])

    def test_wh_word(self):
        text = "Who is coming to the party tonight?"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('WH', tagged_words[0]['tags'])

    def test_indefinite_article(self):
        text = "I saw a bird flying in the sky."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('INDA', tagged_words[2]['tags'])

    def test_accusative_case(self):
        text = "He helped me carry the heavy boxes."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('ACCU', tagged_words[2]['tags'])

    def test_progressive_aspect(self):
        text = "They are studying for their exams right now."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('PGAS', tagged_words[2]['tags'])

    def test_progressive_aspect_end(self):
        text = "Despite the noise, they kept studying."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('PGAS', tagged_words[-2]['tags'])

    def test_comparative(self):
        text = "She is taller than her sister."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('CMADJ', tagged_words[2]['tags'])

    def test_superlative(self):
        text = "This is the best pizza I've ever had."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('SPADJ', tagged_words[3]['tags'])

    def test_non_pos(self):
        text = "Check out the latest news on our website: www.example.com."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn('X', tagged_words[-2]['tags'])


if __name__ == '__main__':
    unittest.main()
