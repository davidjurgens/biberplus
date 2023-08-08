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


if __name__ == '__main__':
    unittest.main()
