import unittest

import spacy

from biberplus.tagger import tag_text


class TestLexicalClassesFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_conj(self):
        text = "tips are a great feature . the wires are slick instead of the iPod 's slightly grippy wires , which"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Instead should be tagged as CONJ
        self.assertIn('CONJ', tagged_words[10]['tags'])

    def test_conj_altogether(self):
        text = "The event was well organized . Altogether, it was a success."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Altogether should be tagged as CONJ when it is preceded by punctuation
        self.assertIn('CONJ', tagged_words[6]['tags'])

    def test_conj_rather(self):
        text = "The movie was okay . Rather, it exceeded my expectations."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Rather should be tagged as CONJ when it is preceded by punctuation
        self.assertIn('CONJ', tagged_words[5]['tags'])

    def test_dwnt(self):
        text = 'a tangent point , and at such a point can only change by an even integer . Thus the multiplicity'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Only should be tagged as a DWNT
        self.assertIn('DWNT', tagged_words[10]['tags'])

    def test_hdg(self):
        text = 'that blow to be borderline . To kayo him and maybe or maybe not kill . You hit again about'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Maybe should be tagged as a HDG
        self.assertIn('HDG', tagged_words[10]['tags'])

    def test_hdg_single_word(self):
        text = 'I think maybe he went to the park.'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Maybe should be tagged as HDG
        self.assertIn('HDG', tagged_words[2]['tags'])

    def test_hdg_two_word(self):
        text = 'It was at about 5pm yesterday'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Kind should be tagged as HDG in the phrase "kind of"
        self.assertIn('HDG', tagged_words[2]['tags'])

    def test_three_word_hdg(self):
        text = 'that blow to be borderline . To kayo him and more or less or maybe not kill . You hit again about'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # More should be tagged as a HDG
        self.assertIn('HDG', tagged_words[10]['tags'])

    def test_amp(self):
        text = 'lie around on the rug during the meal , a very pretty sight as Rob Roy , '
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Very should be tagged as an AMP
        self.assertIn('AMP', tagged_words[10]['tags'])

    def test_emph(self):
        text = 'not be subjected to such a risk , or that such a possibility should not be permitted to endanger the'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Such should be tagged as a EMPH
        self.assertIn('EMPH', tagged_words[10]['tags'])

    def test_emph_does(self):
        text = 'He really does like the cake.'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Does should be tagged as a EMPH
        self.assertIn('EMPH', tagged_words[2]['tags'])

    def test_emph_so(self):
        text = 'She is so happy about the surprise.'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # So should be tagged as a EMPH
        self.assertIn('EMPH', tagged_words[2]['tags'])

    def test_two_word_emph(self):
        text = 'not be subjected to such a risk , or that for sure a possibility should not be permitted to endanger the'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # For should be tagged as a EMPH
        self.assertIn('EMPH', tagged_words[10]['tags'])

    def test_dpar(self):
        text = "were and because we all wanted to be thin . Now , as a woman in her middle 30 's"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Now should be tagged as a DPAR
        self.assertIn('DPAR', tagged_words[10]['tags'])

    def test_demo(self):
        text = "a little bigger than i expected . I just purchased this item and I have not found anywhere on the"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # This should be tagged as DEMO
        self.assertIn('DEMO', tagged_words[10]['tags'])

    def test_demo_this(self):
        text = "a little bigger than i expected . I just purchased this item and I have not found anywhere on the"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # This should be tagged as DEMO
        self.assertIn('DEMO', tagged_words[10]['tags'])

    def test_demo_these(self):
        text = "Those shoes look great. However, these ones seem to be more comfortable."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # These should be tagged as DEMO
        self.assertIn('DEMO', tagged_words[7]['tags'])

    def test_perfect_aspect_with_adverb(self):
        text = "I have really enjoyed the movie."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Have enjoyed should be tagged as PEAS even with intervening adverb
        self.assertIn('PEAS', tagged_words[1]['tags'])

    def test_perfect_aspect_interrogative(self):
        text = "Have you seen the movie?"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Have seen should be tagged as PEAS in questions
        self.assertIn('PEAS', tagged_words[0]['tags'])

    def test_demonstrative_pronoun_with_verb(self):
        text = "This works well."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # This should be tagged as DEMP when followed by verb
        self.assertIn('DEMP', tagged_words[0]['tags'])

    def test_demonstrative_pronoun_with_aux(self):
        text = "Those will arrive tomorrow."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Those should be tagged as DEMP when followed by auxiliary
        self.assertIn('DEMP', tagged_words[0]['tags'])

    def test_that_as_demonstrative(self):
        text = "That is interesting."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # That should be tagged as DEMP when followed by is
        self.assertIn('DEMP', tagged_words[0]['tags'])

    def test_nomz_singular(self):
        text = "The administration made a decision."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Words ending in -tion should be tagged as NOMZ
        self.assertIn('NOMZ', tagged_words[1]['tags'])
        self.assertIn('NOMZ', tagged_words[4]['tags'])

    def test_nomz_plural(self):
        text = "Many suggestions were made."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Plural forms of -tion words should be tagged as NOMZ
        self.assertIn('NOMZ', tagged_words[1]['tags'])

    def test_gerund_long_word(self):
        text = "Understanding complex concepts takes time."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Words >10 chars ending in -ing should be tagged as GER
        self.assertIn('GER', tagged_words[0]['tags'])

    def test_gerund_short_word(self):
        text = "Running is good exercise."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Words <=10 chars ending in -ing should not be tagged as GER
        self.assertNotIn('GER', tagged_words[0]['tags'])


if __name__ == '__main__':
    unittest.main()
