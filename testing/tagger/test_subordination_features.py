import unittest

import stanza

from src.tagger.tagger_utils import build_variable_dictionaries
from src.tagger.word_tagger import WordTagger


class TestSubordinationFeatureFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def test_thvc(self):
        doc = self.pipeline("I 've read a few of these reviews and think that Fisher Price "
                            "must have a quality control issue .")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # That should be tagged as a THVC
        self.assertIn('THVC', tagger.tagged_words[10]['tags'])

    def test_thac(self):
        doc = self.pipeline("twice a day for 20 minutes per use . Disappointing that it failed so quickly "
                            ". I have now owned")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # That should be tagged as a THAC
        self.assertIn('THAC', tagger.tagged_words[10]['tags'])

    def test_whcl(self):
        doc = self.pipeline("it gingerly with his foot . How could anyone know what to do with an assortment "
                            "like that ? Perhaps")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # What should be tagged as WHCL
        self.assertIn('WHCL', tagger.tagged_words[10]['tags'])

    def test_presp(self):
        doc = self.pipeline('practice and for that it is a good resource . Knowing why some aspects '
                            'are not included and having the')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Knowing should be tagged as PRESP
        self.assertIn('PRESP', tagger.tagged_words[10]['tags'])

    def test_pastp(self):
        doc = self.pipeline('. Built in a single week, the house would stand for fifty years')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Built should be tagged as PASTP
        self.assertIn('PASTP', tagger.tagged_words[1]['tags'])

    def test_wzpast(self):
        doc = self.pipeline(
            'in most cases with understanding and restraint . The progress reported by the advisory committee is real . While some')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        for tagged in tagger.tagged_words:
            print(tagged)
        # Reported should be tagged as WZPAST
        self.assertIn('WZPAST', tagger.tagged_words[10]['tags'])

    def test_wzpres(self):
        doc = self.pipeline(
            "and the mean , and he sees the Compson family disintegrating from within . If the barn-burner 's family produces")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Disintegrating should be tagged as WZPAST
        self.assertIn('WZPRES', tagger.tagged_words[10]['tags'])

    def test_tsub(self):
        doc = self.pipeline('we proceed through the seasons of life . Minor characters that surround the love '
                            'triangle are colorful and woven with')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # That should be tagged as a TSUB
        self.assertIn('TSUB', tagger.tagged_words[10]['tags'])

    def test_whsub(self):
        doc = self.pipeline('. There are plenty of reference mentioned at the end which can be followed up '
                            'for more curiosity . Must')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Which should be tagged as a WHSUB
        self.assertIn('WHSUB', tagger.tagged_words[10]['tags'])

    def test_whobj(self):
        doc = self.pipeline("can be brave and courageous . Mafatu is a boy whose mom dies at sea and ever since he was")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Whose should be tagged as a WHOBJ
        self.assertIn('WHOBJ', tagger.tagged_words[10]['tags'])

    def test_pire(self):
        doc = self.pipeline('pencil ! I am a semi-professional singer , one of whose idols is the great '
                            'Judy Garland . No one')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Whose should be tagged as a PIRE
        self.assertIn('PIRE', tagger.tagged_words[11]['tags'])

    def test_sere(self):
        doc = self.pipeline('does not stop until you put the book down , which you will not do '
                            'until you have finished it')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Which should be tagged as a SERE
        self.assertIn('SERE', tagger.tagged_words[10]['tags'])

    def test_tobj(self):
        doc = self.pipeline('the dog that I saw')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # That should be tagged as a TOJB
        self.assertIn('TOBJ', tagger.tagged_words[2]['tags'])

    def test_caus(self):
        doc = self.pipeline(
            'they did also fall under the power of death , because they did eat in disobedience ; and disobedience to')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Because should be tagged as a CAUS
        self.assertIn('CAUS', tagger.tagged_words[10]['tags'])

    def test_conc(self):
        doc = self.pipeline(
            "outsider . When they learn you 're in the hills though , they 'll rally , do n't worry about")
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Though should be tagged as a CONC
        self.assertIn('CONC', tagger.tagged_words[10]['tags'])

    def test_cond(self):
        doc = self.pipeline('so high that the top falls gently over , as if to show that it really is hair and not')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # If should be tagged as a COND
        self.assertIn('COND', tagger.tagged_words[10]['tags'])

    def test_osub(self):
        doc = self.pipeline(
            'his comment on the planter dynasties as they have existed since the decades before the Civil War . It may')
        tagger = WordTagger(doc, self.patterns_dict)
        tagger.run_all()
        # Since should be tagged as an OSUB
        self.assertIn('OSUB', tagger.tagged_words[10]['tags'])



if __name__ == '__main__':
    unittest.main()
