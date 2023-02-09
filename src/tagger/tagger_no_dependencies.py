import inspect


class SimpleWordTagger:
    def __init__(self, sentence, word, word_index, patterns_dict):
        """ Tag a sentence for the Biber tags that have no prior dependencies
            :param sentence: Tagged sentence from stanza
            :param word: Tagged word from stanza
            :param word_index: Position of the word in the sentence
            :param patterns_dict: Dictionary containing list of words for different patterns.
            e.g. public verbs, downtowners, etc.
            :return:
        """

        """ Sentence and word tagged through the stanza library """
        self.sentence = sentence
        self.word = word
        self.word_index = word_index
        self.patterns_dict = patterns_dict
        self.word.tags = []

    def run_all(self):
        """ Run all tagger functions defined in this class"""
        attrs = (getattr(self, name) for name in dir(self))
        methods = filter(inspect.ismethod, attrs)
        for method in methods:
            method()

    def is_first_word_in_sentence(self):
        return self.word_index == 0

    def is_last_word_in_sentence(self):
        return len(self.sentence) == self.word_index + 1

    def is_noun(self):
        return self.word['xpos'][:2] in ['NN', 'PR', 'WP']

    def tag_fpp1(self):
        """ Any item of this list: I, me, us, my, we, our, myself, ourselves """
        if self.word['text'].lower() in self.patterns_dict['first_person_pronouns']:
            self.word.tags.append('FPP1')

    def tag_spp2(self):
        """ Any item of this list: you, your, yourself, yourselves, thy, thee, thyself, thou """
        if self.word['text'].lower() in self.patterns_dict['second_person_pronouns']:
            self.word.tags.append('SPP2')

    def tag_tpp3(self):
        """ Any item of this list: she, he, they, her, him, them, his, their, himself, herself, themselves """
        if self.word['text'].lower() in self.patterns_dict['third_person_pronouns']:
            self.word.tags.append('TPP3')

    def tag_pit(self):
        """ Any pronoun it. Although not specified in Biber (1988), the present program also tags its and
            itself as “Pronoun it”. """
        if self.word['text'].lower() in self.patterns_dict['pronoun_it'] \
                and self.word['xpos'] in ['PRP', 'PRP$', 'WP', 'WP$']:
            self.word.tags.append('PIT')

    def tag_inpr(self):
        """ Any item of this list: anybody, anyone, anything, everybody, everyone, everything, nobody,
            none, nothing, nowhere, somebody, someone, something """
        if self.word['text'].lower() in self.patterns_dict['indefinite_pronouns']:
            self.word.tags.append('INPR')

    def tag_xxo(self):
        """ Analytic negation: word 'not' and to the item n’t_RB"""
        # TODO: Revisit the definition
        if self.word['text'].lower() == 'not' or (self.word['xpos'] == 'RB' and self.word[-3:] == "n't"):
            self.word.tags.append('XXO')

    def tag_syne(self):
        """ Synthetic negation: (no, neither, nor) followed by an adjective, noun, or proper noun"""
        if not self.is_last_word_in_sentence():
            next_word = self.sentence[self.word_index + 1]
            if self.word['text'].lower() in self.patterns_dict['synthetic_negations'] and \
                    next_word in ['xpos'] in ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$']:
                self.word.tags.append('SYNE')

    def tag_place(self):
        """ Any item in the place adverbials list that is not a proper noun (NNP) """
        if self.word['text'].lower() in self.patterns_dict['place_adverbials'] and self.word['xpos'] != "NNP":
            self.word.tags.append('PLACE')

    def tag_pubv(self):
        """ Any item in the public verbs list """
        if self.word['text'].lower() in self.patterns_dict['public_verbs']:
            self.word.tags.append('PUBV')

    def tag_priv(self):
        """ Any item in the private verbs list """
        if self.word['text'].lower() in self.patterns_dict['private_verbs']:
            self.word.tags.append('PRIV')

    def tag_suav(self):
        """ Any item in the suasive verbs list """
        if self.word['text'].lower() in self.patterns_dict['suasive_verbs']:
            self.word.tags.append('SUAV')

    def tag_smp(self):
        """ Any occurrence of the forms of the two verbs seem and appear """
        if self.word['text'].lower() in self.patterns_dict['seem_appear']:
            self.word.tags.append('SMP')

    def tag_cont(self):
        """ Any instance of apostrophe followed by a tagged word OR any instance of the item n’t """
        if ("'" in self.word['text'] and not self.is_last_word_in_sentence()) or (self.word['text'][-3:] == "n't"):
            self.word.tags.append('CONT')

    def tag_dwnt(self):
        """ Any instance of the words in the downtowners list """
        if self.word['text'].lower() in self.patterns_dict['downtowners']:
            self.word.tags.append('DWNT')

    def tag_amp(self):
        """ Any instance of the items in the amplifiers list """
        if self.word['text'].lower() in self.patterns_dict['amplifiers']:
            self.word.tags.append('AMP')

    def tag_rb(self):
        """ Any adverb i.e. POS tags RB, RBS, RBR, WRB"""
        if self.word['xpos'] in ['RB', 'RBS', 'RBR', 'WRB']:
            self.word.tags.append('RB')

    def tag_caus(self):
        """ Any occurrence of the word because """
        if self.word['text'].lower() == 'because':
            self.word.tags.append('CAUS')

    def tag_conc(self):
        """ Any occurrence of the words although, though, tho """
        if self.word['text'].lower() in ['although', 'though', 'tho']:
            self.word.tags.append('CONC')

    def tag_cond(self):
        """ Any occurrence of the words if or unless"""
        if self.word['text'].lower() in ['if', 'unless']:
            self.word.tags.append('COND')

    def tag_vbd(self):
        """ Past tense POS """
        if self.word['xpos'] == 'VBD':
            self.word.tags.append('VBD')

    def tag_ex(self):
        """ Existential there from the POS tags"""
        if self.word['xpos'] == 'EX':
            self.word.tags.append('EX')

    def tag_dpar(self):
        """ Discourse particle: the words well, now, anyhow, anyways preceded by a punctuation mark """
        if not self.is_first_word_in_sentence():
            prev_word = self.sentence[self.word_index - 1]
            if self.word['text'].lower() in self.patterns_dict['discourse_particles'] and prev_word['upos'] == 'PUNC':
                self.word.tags.append('DPAR')

    def tag_pomd(self):
        """ The possibility modals listed by Biber (1988): can, may, might, could """
        if self.word['text'].lower() in self.patterns_dict['possibility_modals']:
            self.word.tags.append('POMD')

    def tag_nemd(self):
        """ The necessity modals listed by Biber (1988): ought, should, must. """
        if self.word['text'].lower() in self.patterns_dict['necessity_modals']:
            self.word.tags.append('NEMD')

    def tag_conj(self):
        """ Conjucts finds any item in the conjucts list with preceding punctuation.
        Only the first word is tagged """
        if not self.is_first_word_in_sentence():
            prev_word = self.sentence[self.word_index - 1]
            if prev_word['upos'] == 'PUNC' and self.word['text'].lower() in self.patterns_dict['conjucts']:
                self.word.tags.append('CONJ')

    def tag_ger(self):
        """ Gerunds with length >= 10 are nominal form (N) that ends in –ing or –ings """
        if len(self.word['text']) >= 10 and \
                (self.word['text'].lower()[-3:] == 'ing' or self.word['text'].lower()[-4:] == 'ings') and \
                (self.is_noun()):
            self.word.tags.append('GER')

    def tag_vprt(self):
        """ Present tense: VBP or VBZ tag"""
        if self.word['xpos'] in ['VBP', 'VBZ']:
            self.word.tags.append('VPRT')

    def tag_time(self):
        """ Time adverbials with the exception: soon is not a time adverbial if it is followed by the word as """
        if self.word['text'].lower() in self.patterns_dict['time_adverbials']:
            if self.word['text'].lower() != 'soon':
                self.word.tags.append('TIME')

            if not self.is_last_word_in_sentence():
                next_word = self.sentence[self.word_index + 1]
                if next_word['text'].lower() != 'as':
                    self.word.tags.append('TIME')

    def tag_to(self):
        """ Infinitives: POS tag TO that are not a preposition """
        if self.word['xpos'] == 'TO':
            # Prepositions are 'to's followed by
            filter_preps = ['IN', 'CD', 'DT', 'JJ', 'PRP$', 'WP$', 'WDT', 'WP', 'WRB',
                            'PDT', 'N', 'NNS', 'NP', 'NPs', 'PRP']
            if not self.is_last_word_in_sentence():
                next_word = self.sentence[self.word_index + 1]
                if next_word['xpos'] not in filter_preps:
                    self.word.tags.append('TO')
