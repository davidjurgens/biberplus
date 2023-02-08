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

    def tag_analytic_negation(self):
        if self.word['text'].lower() == 'not' or (self.word['xpos'] == 'RB' and self.word[-3:] == "n't"):
            self.word.tags.append('XXO')

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
        if ("'" in self.word['text'] and len(self.sentence) > self.word_index) or (self.word['text'][-3:] == "n't"):
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
