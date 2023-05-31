import unittest

import spacy

from bibermda.tagger import tag_text


class TestSubordinationFeatureFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])

    def test_thvc(self):
        text = "I've read a few of these reviews and think that Fisher Price must have a quality control issue ."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # That should be tagged as a THVC
        self.assertIn('THVC', tagged_words[10]['tags'])

    def test_thac(self):
        text = "twice a day for 20 minutes per use . Disappointing that it failed so quickly . I have now owned"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # That should be tagged as a THAC
        self.assertIn('THAC', tagged_words[10]['tags'])

    def test_whcl(self):
        text = "it gingerly with his foot . How could anyone know what to do with an assortment like that ? Perhaps"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # What should be tagged as WHCL
        self.assertIn('WHCL', tagged_words[10]['tags'])

    def test_presp(self):
        text = 'practice and for that it is a good resource . Knowing why some aspects are not included and having the'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Knowing should be tagged as PRESP
        self.assertIn('PRESP', tagged_words[10]['tags'])

    def test_pastp(self):
        text = '. Built in a single week, the house would stand for fifty years'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Built should be tagged as PASTP
        self.assertIn('PASTP', tagged_words[1]['tags'])

    def test_wzpast(self):
        text = 'in most cases with understanding and restraint . The progress reported by the advisory ' \
               'committee is real . While some'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Reported should be tagged as WZPAST
        self.assertIn('WZPAST', tagged_words[10]['tags'])

    def test_wzpres(self):
        text = "and the mean , and he sees the Compson family disintegrating from within . If the barn-burner 's " \
               "family produces"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Disintegrating should be tagged as WZPAST
        self.assertIn('WZPRES', tagged_words[10]['tags'])

    def test_tsub(self):
        text = 'we proceed through the seasons of life . Minor characters that surround the love ' \
               'triangle are colorful and woven with'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # That should be tagged as a TSUB
        self.assertIn('TSUB', tagged_words[10]['tags'])

    def test_whsub(self):
        text = '. There are plenty of reference mentioned at the end which can be followed ' \
               'up for more curiosity . Must'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Which should be tagged as a WHSUB
        self.assertIn('WHSUB', tagged_words[10]['tags'])

    def test_whobj(self):
        text = "can be brave and courageous . Mafatu is a boy whose mom dies at sea and ever since he was"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Whose should be tagged as a WHOBJ
        self.assertIn('WHOBJ', tagged_words[10]['tags'])

    def test_pire(self):
        text = 'pencil ! I am a semi-professional singer , one of whose idols is the great Judy Garland . No one'

        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Whose should be tagged as a PIRE
        self.assertIn('PIRE', tagged_words[12]['tags'])

    def test_sere(self):
        text = 'does not stop until you put the book down , which you will not do until you have finished it'

        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Which should be tagged as a SERE
        self.assertIn('SERE', tagged_words[10]['tags'])

    def test_tobj(self):
        text = 'the dog that I saw'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # That should be tagged as a TOJB
        self.assertIn('TOBJ', tagged_words[2]['tags'])

    def test_caus(self):
        text = 'they did also fall under the power of death , because they did eat in disobedience ; and disobedience to'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Because should be tagged as a CAUS
        self.assertIn('CAUS', tagged_words[10]['tags'])

    def test_conc(self):
        text = "outsider . When they learn you 're in the hills though , they 'll rally , do n't worry about"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Though should be tagged as a CONC
        self.assertIn('CONC', tagged_words[10]['tags'])

    def test_cond(self):
        text = 'so high that the top falls gently over , as if to show that it really is hair and not'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # If should be tagged as a COND
        self.assertIn('COND', tagged_words[10]['tags'])

    def test_osub(self):
        text = 'his comment on the planter dynasties as they have existed since the decades before the \
        Civil War . It may'
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Since should be tagged as an OSUB
        self.assertIn('OSUB', tagged_words[10]['tags'])


if __name__ == '__main__':
    unittest.main()
