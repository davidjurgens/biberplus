class TagHelper:

    def __init__(self, patterns):
        self.patterns = patterns

    @staticmethod
    def is_adjective(word):
        return word and word['upos'] == 'ADJ'

    @staticmethod
    def is_adposition(word):
        return word and word['upos'] == 'ADP'

    @staticmethod
    def is_adverb(word):
        return word and (word['upos'] == 'ADV' or 
                        word['xpos'] in ['RB', 'RBR', 'RBS', 'WRB'])

    @staticmethod
    def is_auxiliary(word):
        return word and word['upos'] == 'AUX'

    @staticmethod
    def is_coordinating_conjunction(word):
        return word and word['upos'] == 'CCONJ'

    @staticmethod
    def is_determiner(word):
        return word and word['upos'] == 'DET'

    @staticmethod
    def is_interjection(word):
        return word and word['upos'] == 'INTJ'

    @staticmethod
    def is_noun(word):
        return word and word['upos'] == 'NOUN'

    @staticmethod
    def is_numeral(word):
        return word and word['upos'] == 'NUM'

    @staticmethod
    def is_particle(word):
        return word and word['upos'] == 'PART'

    @staticmethod
    def is_pronoun(word):
        return word and word['upos'] == 'PRON'

    @staticmethod
    def is_proper_noun(word):
        return word and word['upos'] == 'PROPN'

    @staticmethod
    def is_punctuation(word):
        return word and word['upos'] == 'PUNCT'

    @staticmethod
    def is_subordinating_conjunction(word):
        return word and word['upos'] == 'SCONJ'

    @staticmethod
    def is_symbol(word):
        return word and word['upos'] == 'SYM'

    @staticmethod
    def is_verb(word):
        return word and word['upos'] == 'VERB'

    @staticmethod
    def is_any_noun(word):
        return word and word['xpos'].startswith('NN')

    @staticmethod
    def is_any_verb(word):
        return word['upos'] == 'VERB' or word['upos'] == 'AUX'

    @staticmethod
    def is_past_tense(word):
        return "Tense=Past" in word['feats']

    @staticmethod
    def is_indefinite_article(word):
        return "Definite=Ind" in word['feats']

    @staticmethod
    def is_infinitive(word):
        return "VerbForm=Inf" in word['feats']

    @staticmethod
    def is_possesive_pronoun(word):
        return word['xpos'] == 'PRP$' or word['xpos'] == 'WP$'

    @staticmethod
    def is_article(word):
        return "PronType=Art" in word['feats']

    @staticmethod
    def is_subject_pronoun(word):
        return "PronType=Prs" in word['feats']

    @staticmethod
    def is_accusative_case(word):
        return "Case=Acc" in word['feats']

    @staticmethod
    def is_progressive_aspect(word):
        return "Aspect=Prog" in word['feats']

    @staticmethod
    def is_comparative_adjective(word):
        return "Degree=Cmp" in word['feats']

    @staticmethod
    def is_superlative_adjective(word):
        return "Degree=Sup" in word['feats']

    @staticmethod
    def is_non_pos(word):
        return word['upos'] == 'X'

    def is_quantifier(self, word):
        return word and word['text'].lower() in self.patterns['quantifiers']

    def is_indefinite_pronoun(self, word):
        return word and word['text'].lower() in self.patterns['indefinite_pronouns']

    def is_quantifier_pronoun(self, word):
        return word and word['text'].lower() in self.patterns['quantifier_pronouns']

    def is_preposition(self, word):
        return word and (word['upos'] == 'ADP' or word['xpos'] == 'IN' or 
                        word['text'].lower() in self.patterns['prepositional_phrases'])

    def is_be(self, word):
        return word and word['text'].lower() in self.patterns['be']

    def is_do(self, word):
        return word and word['text'].lower() in self.patterns['do']

    def is_have(self, word):
        return word and word['text'].lower() in self.patterns['have']
