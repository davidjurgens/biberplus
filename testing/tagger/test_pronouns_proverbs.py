import unittest

import spacy

from biberplus.tagger import tag_text


class TestPronounProverbFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_fpp1(self):
        text = 'the soil soft during these early days of growth . I like sawdust for this , or hay . When'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # I should be tagged as a FPP1
        self.assertIn('FPP1', tagged_words[10]['tags'])

    def test_spp2(self):
        text = ". . -RRB- The new interpretation makes sense though if you think about it . " \
               "By the way , he"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # You should be tagged as a SPP2
        self.assertIn('SPP2', tagged_words[10]['tags'])

    def test_tpp3(self):
        text = 'a child till he was sixteen , a youth till he was five-and-twenty , and a young man till he'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # He should be tagged as a TPP3
        self.assertIn('TPP3', tagged_words[10]['tags'])

    def test_pit(self):
        text = 'sometimes answers itself , and that the way in which it is posed frequently shapes ' \
               'the answer . Chewing it'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # It should be tagged as a PIT
        self.assertIn('PIT', tagged_words[10]['tags'])

    def test_demp(self):
        text = 'Vernon on the morning of the regular tallyho run . This was an honor , ' \
               'like dining with a captain'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # This should be tagged as a DEMP
        self.assertIn('DEMP', tagged_words[10]['tags'])

        text = 'Vernon on the morning of the regular tallyho run . This was an honor , ' \
               'like dining with a captain'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # "This" should be tagged as a DEMP
        self.assertIn('DEMP', tagged_words[10]['tags'])

        text = 'Those are the moments I cherish the most. And these inspire me to strive harder.'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # "Those" should be tagged as a DEMP
        self.assertIn('DEMP', tagged_words[0]['tags'])

    def test_inpr(self):
        text = "I turned away from her coldly . `` It was nobody 's fault . She overplayed her hand '' ."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Nobody should be tagged as a INPR
        self.assertIn('INPR', tagged_words[11]['tags'])

    def test_prod(self):
        text = "Whom do they seek?"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # The "do" shouldn't be tagged as a PROD because it's preceded by a WH pronoun
        self.assertNotIn('PROD', tagged_words[1]['tags'])

    def test_fpp1_variants(self):
        """Test different forms of first person pronouns"""
        texts = [
            ("We need to finish this project.", "We"),
            ("This belongs to me and my friend.", "me"),
            ("Our team worked hard on this.", "Our"),
            ("I did this task myself.", "myself"),
            ("Let us handle this situation.", "us"),
            ("All of ourselves were present.", "ourselves")
        ]
        
        for text, target_word in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            word_index = next(i for i, w in enumerate(tagged_words) if w['text'] == target_word)
            self.assertIn('FPP1', tagged_words[word_index]['tags'])

    def test_spp2_variants(self):
        """Test different forms of second person pronouns"""
        texts = [
            ("You should try this approach.", "You"),
            ("Is this your book?", "your"),
            ("You can do it yourself.", "yourself"),
            ("All of yourselves must attend.", "yourselves"),
            # Test archaic forms
            ("Thou art wise.", "Thou"),
            ("This is thy choice.", "thy"),
            ("Speak thee truth.", "thee"),
            ("Compose thyself.", "thyself")
        ]
        
        for text, target_word in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            word_index = next(i for i, w in enumerate(tagged_words) if w['text'] == target_word)
            self.assertIn('SPP2', tagged_words[word_index]['tags'])

    def test_tpp3_variants(self):
        """Test different forms of third person pronouns"""
        texts = [
            ("She went to the store.", "She"),
            ("He is working late.", "He"),
            ("They arrived early.", "They"),
            ("Give it to her.", "her"),
            ("This belongs to him.", "him"),
            ("The book is their property.", "their"),
            ("He did it himself.", "himself"),
            ("She wrote it herself.", "herself"),
            ("They did it themselves.", "themselves")
        ]
        
        for text, target_word in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            word_index = next(i for i, w in enumerate(tagged_words) if w['text'] == target_word)
            self.assertIn('TPP3', tagged_words[word_index]['tags'])

    def test_pit_variants(self):
        """Test different forms of pronoun it"""
        texts = [
            ("It seems to work fine.", "It"),
            ("The car and its engine.", "its"),
            ("The machine fixed itself.", "itself")
        ]
        
        for text, target_word in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            word_index = next(i for i, w in enumerate(tagged_words) if w['text'] == target_word)
            self.assertIn('PIT', tagged_words[word_index]['tags'])

    def test_demp_patterns(self):
        """Test demonstrative pronouns in different patterns"""
        texts = [
            # Followed by verb
            ("This works perfectly.", "This"),
            # Followed by auxiliary verb
            ("Those are working well.", "Those"),
            # Followed by punctuation
            ("I like these.", "these"),
            # Followed by WH pronoun
            ("This which I saw.", "This"),
            # Followed by 'and'
            ("These and those.", "These")
        ]
        
        for text, target_word in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            word_index = next(i for i, w in enumerate(tagged_words) if w['text'] == target_word)
            self.assertIn('DEMP', tagged_words[word_index]['tags'])

    def test_inpr_variants(self):
        """Test different indefinite pronouns"""
        texts = [
            ("Anybody can join us.", "Anybody"),
            ("Everyone was present.", "Everyone"),
            ("Nothing was found.", "Nothing"),
            ("Somebody called earlier.", "Somebody"),
            ("Everything is ready.", "Everything"),
            ("Nobody answered.", "Nobody"),
            ("Something strange happened.", "Something"),
            ("There was none left.", "none"),
            ("They went nowhere.", "nowhere")
        ]
        
        for text, target_word in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            word_index = next(i for i, w in enumerate(tagged_words) if w['text'] == target_word)
            self.assertIn('INPR', tagged_words[word_index]['tags'])

    def test_prod_patterns(self):
        """Test pro-verb do patterns"""
        texts_should_be_prod = [
            "I did the homework.",
            "What did you do today?",
            "They do good work."
        ]
        
        texts_should_not_be_prod = [
            "Do you know the answer?",  # Auxiliary DO
            "What do they want?",       # DO after WH word
            "How do they manage?",      # DO after WH word
            "Do not go there."          # Auxiliary DO with negation
        ]
        
        for text in texts_should_be_prod:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            prod_tokens = [
                w for w in tagged_words
                if w['text'].lower() in ['do', 'does', 'did', 'done'] and 'PROD' in w['tags']
            ]
            self.assertTrue(len(prod_tokens) > 0,
                            f"Expected PROD tag in one of the 'do' tokens in: {text}")
        
        for text in texts_should_not_be_prod:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            prod_tokens = [
                w for w in tagged_words
                if w['text'].lower() in ['do', 'does', 'did', 'done'] and 'PROD' in w['tags']
            ]
            self.assertEqual(len(prod_tokens), 0, f"Did not expect PROD tag in any 'do' token in: {text}")


if __name__ == '__main__':
    unittest.main()
