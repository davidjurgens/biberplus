class TagHelper:

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
    def is_future_tense(word):
        return "Tense=Past"

    @staticmethod
    def is_quantifier(word, patterns_dict):
        return word['text'].lower() in patterns_dict['quantifiers']

    @staticmethod
    def is_indefinite_pronoun(word, patterns_dict):
        return word['text'].lower() in patterns_dict['indefinite_pronouns']

    @staticmethod
    def is_quantifier_pronoun(word, patterns_dict):
        return word['text'].lower() in patterns_dict['quantifier_pronouns']
