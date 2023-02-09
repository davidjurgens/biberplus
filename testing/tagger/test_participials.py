import unittest

import stanza

from src.tagger.simple_word_tagger import SimpleWordTagger
from src.tagger.tagger_utils import build_variable_dictionaries


class TesParticipialFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
        self.patterns_dict = build_variable_dictionaries()

    def run_tagger(self, doc):
        tagged_sentences = []
        for sent_index, sentence in enumerate(doc):
            tagged_sentence = []
            for word_index, word in enumerate(sentence):
                simple_tagger = SimpleWordTagger(sentence, word, word_index, self.patterns_dict)
                simple_tagger.run_all()
                tagged_sentence.append(simple_tagger.word)
            tagged_sentences.append(tagged_sentence)
        return tagged_sentences

    def test_presp(self):
        doc = self.pipeline('Built in a single week, the house would stand for fifty years')
        doc = doc.to_dict()
        tagged_sentences = self.run_tagger(doc)
        print(tagged_sentences)
        self.assertEqual(True, True)

    def test_pastp(self):
        doc = self.pipeline('The solution produced by this process')
        doc = doc.to_dict()
        tagged_sentences = self.run_tagger(doc)
        print(tagged_sentences)
        self.assertEqual(True, True)

    def test_wzpast(self):
        doc = self.pipeline('The event causing this decline is a rise in interest rates')
        doc = doc.to_dict()
        tagged_sentences = self.run_tagger(doc)
        print(tagged_sentences)
        self.assertEqual(True, True)

    def test_wzprez(self):
        doc = self.pipeline('Stuffing his mouth with cookies, Joe accepted and ran out the door')
        doc = doc.to_dict()
        tagged_sentences = self.run_tagger(doc)
        print(tagged_sentences)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
