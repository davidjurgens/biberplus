class TagHelper:

    def __init__(self, patterns):
        self.patterns = patterns

    @staticmethod
    def is_first_word(word):
        return word['id'] == 1

    @staticmethod
    def is_adjective(word):
        return word['upos'] == 'ADJ'

    @staticmethod
    def is_adposition(word):
        return word['upos'] == 'ADP'

    @staticmethod
    def is_adverb(word):
        return word['upos'] == 'ADV'

    @staticmethod
    def is_auxiliary(word):
        return word['upos'] == 'AUX'

    @staticmethod
    def is_coordinating_conjunction(word):
        return word['upos'] == 'CCONJ'

    @staticmethod
    def is_determiner(word):
        return word['upos'] == 'DET'

    @staticmethod
    def is_interjection(word):
        return word['upos'] == 'INTJ'

    @staticmethod
    def is_noun(word):
        return word['upos'] == 'NOUN'

    @staticmethod
    def is_numeral(word):
        return word['upos'] == 'NUM'

    @staticmethod
    def is_particle(word):
        return word['upos'] == 'PART'

    @staticmethod
    def is_pronoun(word):
        return word['upos'] == 'PRON'

    @staticmethod
    def is_proper_noun(word):
        return word['upos'] == 'PROPN'

    @staticmethod
    def is_punctuation(word):
        return word['upos'] == 'PUNCT'

    @staticmethod
    def is_subordinating_conjunction(word):
        return word['upos'] == 'SCONJ'

    @staticmethod
    def is_symbol(word):
        return word['upos'] == 'SYM'

    @staticmethod
    def is_verb(word):
        return word['upos'] == 'VERB'

    @staticmethod
    def is_past_tense(word):
        return "Tense=Past" in word['feats']

    @staticmethod
    def is_present_tense(word):
        return "Tense=Fut" in word['feats']

    @staticmethod
    def is_possesive_pronoun(word):
        return word['upos'] == 'PRON' and 'feats' in word and 'Poss=Yes' in word['feats']

    @staticmethod
    def is_future_tense(word):
        return "Tense=Past"

    def is_quantifier(self, word):
        return word['text'].lower() in self.patterns['quantifiers']

    def is_indefinite_pronoun(self, word):
        return word['text'].lower() in self.patterns['indefinite_pronouns']

    def is_quantifier_pronoun(self, word):
        return word['text'].lower() in self.patterns['quantifier_pronouns']

    def is_preposition(self, word):
        return word['text'].lower() in self.patterns['prepositional_phrases']
    def is_be(self, word):
        return word['text'].lower() in self.patterns['be']

    def is_do(self, word):
        return word['text'].lower() in self.patterns['do']

    def is_have(self, word):
        return word['text'].lower() in self.patterns['have']
