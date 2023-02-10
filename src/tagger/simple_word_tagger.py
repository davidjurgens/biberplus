import inspect


class SimpleWordTagger:
    def __init__(self, doc, patterns_dict):
        """ Tag a sentence for the Biber tags that have no prior dependencies
            :param doc: Tagged sentence from stanza
            :param patterns_dict: Dictionary containing list of words for different patterns.
            e.g. public verbs, downtoners, etc.
            :return:
        """
        self.doc = doc
        self.words = list(doc.iter_words())
        self.tagged_words = []
        self.patterns_dict = patterns_dict

    def run_all(self):
        """ Run all tagger functions defined in this class"""
        attrs = (getattr(self, name) for name in dir(self))

        # Get all the methods that start with 'tag'
        methods = filter(inspect.ismethod, attrs)
        tag_methods = [m for m in methods if m.__name__[:3] == 'tag']

        for index, word in enumerate(self.words):
            tagged_word = word.to_dict()
            tagged_word['tags'] = []
            for tag_method in tag_methods:
                tag = tag_method(tagged_word, index)
                if tag:
                    tagged_word['tags'].append(tag)
            self.tagged_words.append(tagged_word)

    def get_previous_word(self, index):
        return self.words[index - 1].to_dict()

    def get_next_word(self, index):
        if index + 1 < len(self.words):
            return self.words[index + 1].to_dict()

    @staticmethod
    def is_first_word_in_sentence(word):
        return word['id'] == 1

    def is_last_word_in_sentence(self, word_index):
        next_word = self.get_next_word(word_index)
        return word_index + 1 == len(self.words) or (next_word and self.is_first_word_in_sentence(next_word))

    @staticmethod
    def is_noun(word):
        return word['upos'] == 'NOUN'

    def tag_fpp1(self, word, word_index):
        """ Any item of this list: I, me, us, my, we, our, myself, ourselves """
        if word['text'].lower() in self.patterns_dict['first_person_pronouns']:
            return 'FPP1'

    def tag_spp2(self, word, word_index):
        """ Any item of this list: you, your, yourself, yourselves, thy, thee, thyself, thou """
        if word['text'].lower() in self.patterns_dict['second_person_pronouns']:
            return 'SPP2'

    def tag_tpp3(self, word, word_index):
        """ Any item of this list: she, he, they, her, him, them, his, their, himself, herself, themselves """
        if word['text'].lower() in self.patterns_dict['third_person_pronouns']:
            return 'TPP3'

    def tag_pit(self, word, word_index):
        """ Any pronoun it. Although not specified in Biber (1988), the present program also tags its and
            itself as “Pronoun it”. """
        if word['text'].lower() in self.patterns_dict['pronoun_it'] \
                and word['xpos'] in ['PRP', 'PRP$', 'WP', 'WP$']:
            return 'PIT'

    def tag_inpr(self, word, word_index):
        """ Any item of this list: anybody, anyone, anything, everybody, everyone, everything, nobody,
            none, nothing, nowhere, somebody, someone, something """
        if word['text'].lower() in self.patterns_dict['indefinite_pronouns']:
            return 'INPR'

    def tag_xxo(self, word, word_index):
        """ Analytic negation: word 'not' and to the item n’t_RB"""
        # TODO: Revisit the definition
        if word['text'].lower() == 'not' or (word['xpos'] == 'RB' and word['text'][-3:].lower() == "n't"):
            return 'XXO'

    def tag_syne(self, word, word_index):
        """ Synthetic negation: (no, neither, nor) followed by an adjective, noun, or proper noun"""
        if not self.is_last_word_in_sentence(word_index):
            next_word = self.get_next_word(word_index)
            if next_word:
                if word['text'].lower() in self.patterns_dict['synthetic_negations'] and \
                        next_word['xpos'] in ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$']:
                    return 'SYNE'

    def tag_place(self, word, word_index):
        """ Any item in the place adverbials list that is not a proper noun (NNP) """
        if word['text'].lower() in self.patterns_dict['place_adverbials'] and word['xpos'] != "NNP":
            return 'PLACE'

    def tag_pubv(self, word, word_index):
        """ Any item in the public verbs list """
        if word['text'].lower() in self.patterns_dict['public_verbs']:
            return 'PUBV'

    def tag_priv(self, word, word_index):
        """ Any item in the private verbs list """
        if word['text'].lower() in self.patterns_dict['private_verbs']:
            return 'PRIV'

    def tag_suav(self, word, word_index):
        """ Any item in the suasive verbs list """
        if word['text'].lower() in self.patterns_dict['suasive_verbs']:
            return 'SUAV'

    def tag_smp(self, word, word_index):
        """ Any occurrence of the forms of the two verbs seem and appear """
        if word['text'].lower() in self.patterns_dict['seem_appear']:
            return 'SMP'

    def tag_cont(self, word, word_index):
        """ Any instance of apostrophe followed by a tagged word OR any instance of the item n’t """
        if ("'" in word['text'] and not self.is_last_word_in_sentence(word_index)) or (
                word['text'][-3:].lower() == "n't"):
            return 'CONT'

    def tag_dwnt(self, word, word_index):
        """ Any instance of the words in the downtowners list """
        if word['text'].lower() in self.patterns_dict['downtoners']:
            return 'DWNT'

    def tag_amp(self, word, word_index):
        """ Any instance of the items in the amplifiers list """
        if word['text'].lower() in self.patterns_dict['amplifiers']:
            return 'AMP'

    def tag_rb(self, word, word_index):
        """ Any adverb i.e. POS tags RB, RBS, RBR, WRB"""
        if word['xpos'] in ['RB', 'RBS', 'RBR', 'WRB']:
            return 'RB'

    def tag_caus(self, word, word_index):
        """ Any occurrence of the word because """
        if word['text'].lower() == 'because':
            return 'CAUS'

    def tag_conc(self, word, word_index):
        """ Any occurrence of the words although, though, tho """
        if word['text'].lower() in ['although', 'though', 'tho']:
            return 'CONC'

    def tag_cond(self, word, word_index):
        """ Any occurrence of the words if or unless"""
        if word['text'].lower() in ['if', 'unless']:
            return 'COND'

    def tag_vbd(self, word, word_index):
        """ Past tense POS """
        if word['xpos'] == 'VBD':
            return 'VBD'

    def tag_ex(self, word, word_index):
        """ Existential there from the POS tags"""
        if word['xpos'] == 'EX':
            return 'EX'

    def tag_dpar(self, word, word_index):
        """ Discourse particle: the words well, now, anyhow, anyways preceded by a punctuation mark """
        if not self.is_first_word_in_sentence(word):
            prev_word = self.get_previous_word(word_index)
            if word['text'].lower() in self.patterns_dict['discourse_particles'] and prev_word['upos'] == 'PUNC':
                return 'DPAR'

    def tag_pomd(self, word, word_index):
        """ The possibility modals listed by Biber (1988): can, may, might, could """
        if word['text'].lower() in self.patterns_dict['possibility_modals']:
            return 'POMD'

    def tag_nemd(self, word, word_index):
        """ The necessity modals listed by Biber (1988): ought, should, must. """
        if word['text'].lower() in self.patterns_dict['necessity_modals']:
            return 'NEMD'

    def tag_conj(self, word, word_index):
        """ Conjucts finds any item in the conjucts list with preceding punctuation.
        Only the first word is tagged """
        if not self.is_first_word_in_sentence(word):
            prev_word = self.get_previous_word(word_index)
            if prev_word['upos'] == 'PUNC' and word['text'].lower() in self.patterns_dict['conjucts']:
                return 'CONJ'

    def tag_ger(self, word, word_index):
        """ Gerunds with length >= 10 are nominal form (N) that ends in –ing or –ings """
        if len(word['text']) >= 10 and \
                (word['text'].lower()[-3:] == 'ing' or word['text'].lower()[-4:] == 'ings') and \
                (self.is_noun(word)):
            return 'GER'

    def tag_vprt(self, word, word_index):
        """ Present tense: VBP or VBZ tag"""
        if word['xpos'] in ['VBP', 'VBZ']:
            return 'VPRT'

    def tag_time(self, word, word_index):
        """ Time adverbials with the exception: soon is not a time adverbial if it is followed by the word as """
        if word['text'].lower() in self.patterns_dict['time_adverbials']:
            if word['text'].lower() != 'soon':
                return 'TIME'

            if not self.is_last_word_in_sentence(word_index):
                next_word = self.get_next_word(word_index)
                if next_word and next_word['text'].lower() != 'as':
                    return 'TIME'

    def tag_to(self, word, word_index):
        """ Infinitives: POS tag TO that are not a preposition """
        if word['xpos'] == 'TO':
            # Prepositions are 'to's followed by
            filter_preps = ['IN', 'CD', 'DT', 'JJ', 'PRP$', 'WP$', 'WDT', 'WP', 'WRB',
                            'PDT', 'N', 'NNS', 'NP', 'NPs', 'PRP']
            if not self.is_last_word_in_sentence(word_index):
                next_word = self.get_next_word(word_index)
                if next_word and next_word['xpos'] not in filter_preps:
                    return 'TO'

    def tag_presp(self, word, word_index):
        """ Present participial clauses: a punctuation mark is followed by a present participial
        form of a verb (VBG) followed by a preposition (PIN), a determiner (DT, QUAN, CD), a WH pronoun,
        a WH possessive pronoun (WP$), any WH word, any pronoun (PRP) or any adverb (RB)-"""
        pass

    def tag_pastp(self, word, word_index):
        """ Past partcipial clauses: punctuation followed by VBN -> PIN or RB
         e.g. 'Built' in a single week, the house would stand for fifty years"""
        pass

    def tag_wzpast(self, word, word_index):
        """ Past participial WHIZ deletion relatives: a noun (N) or quantifier pronoun (QUPR) followed by a
        past participial form of a verb (VBN) followed by a preposition (PIN) or an adverb (RB) or a form of BE.
        e.g. The solution 'produced' by this process """
        pass

    def tag_wzpres(self, word, word_index):
        """ Present participial WHIZ deletion relatives: VBG preceded by an NN
        e.g. the 'causing' this decline' is """
        pass
