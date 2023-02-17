import inspect

import numpy as np

from src.tagger.tag_helper import TagHelper


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
        self.patterns_dict = patterns_dict
        self.helper = TagHelper()
        self.tagged_words = []
        self.word_lengths = []
        self.adverb_count = 0
        self.mean_word_length = -1

    def run_all(self):
        """ Run all tagger functions defined in this class"""
        attrs = (getattr(self, name) for name in dir(self))

        # Get all the methods that start with 'tag'
        methods = filter(inspect.ismethod, attrs)
        tag_methods = [m for m in methods if m.__name__[:3] == 'tag']

        for index, word in enumerate(self.words):
            tagged_word = word.to_dict()
            tagged_word[index] = index
            tagged_word['tags'] = []
            for tag_method in tag_methods:
                tag = tag_method(tagged_word, index)
                if tag:
                    tagged_word['tags'].append(tag)

            self.tagged_words.append(tagged_word)

        self.mean_word_length = np.array(self.word_lengths).mean()

    def update_doc_level_stats(self, word):
        if self.helper.is_adverb(word):
            self.adverb_count += 1

        self.word_lengths.append(len(word['text']))

    def get_previous_word(self, index):
        return self.words[index - 1].to_dict()

    def get_next_word(self, index):
        if index + 1 < len(self.words):
            return self.words[index + 1].to_dict()

    def is_last_word_in_sentence(self, word_index):
        next_word = self.get_next_word(word_index)
        return word_index + 1 == len(self.words) or (next_word and self.helper.is_first_word(next_word))

    def tag_fpp1(self, word, word_index):
        """ Any item of this list: I, me, us, my, we, our, myself, ourselves and their contractions.
        Tokenizer separates contractionas """
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
        if self.helper.is_pronoun(word) and word['text'].lower() in self.patterns_dict['pronoun_it']:
            return 'PIT'

    def tag_inpr(self, word, word_index):
        """ Any item of this list: anybody, anyone, anything, everybody, everyone, everything, nobody,
            none, nothing, nowhere, somebody, someone, something """
        if word['text'].lower() in self.patterns_dict['indefinite_pronouns']:
            return 'INPR'

    def tag_xxo(self, word, word_index):
        """ Analytic negation: word 'not' and to the item n’t_RB"""
        if word['text'].lower() in self.patterns_dict['analytic_negation']:
            return 'XXO'

    def tag_syne(self, word, word_index):
        """ Synthetic negation: (no, neither, nor) followed by an adjective, noun, or proper noun"""
        if not self.is_last_word_in_sentence(word_index):
            next_word = self.get_next_word(word_index)
            if next_word:
                if word['text'].lower() in self.patterns_dict['synthetic_negations'] and (
                        self.helper.is_adjective(next_word) or self.helper.is_noun(next_word)
                        or self.helper.is_proper_noun(word)):
                    return 'SYNE'

    def tag_place(self, word, word_index):
        """ Any item in the place adverbials list that is not a proper noun (NNP) """
        if word['text'].lower() in self.patterns_dict['place_adverbials'] and not self.helper.is_proper_noun(word):
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
        if (word['text'][-3:].lower() == "n't"
                or word['text'][0] == "'" and len(word['text']) > 1):
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
        if self.helper.is_adverb(word):
            return 'RB'

    def tag_caus(self, word, word_index):
        """ Any occurrence of the word because """
        if word['text'].lower() == 'because':
            return 'CAUS'

    def tag_conc(self, word, word_index):
        """ Any occurrence of the words although, though, tho """
        if word['text'].lower() in self.patterns_dict['concessive_adverbial_subordinators']:
            return 'CONC'

    def tag_cond(self, word, word_index):
        """ Any occurrence of the words if or unless"""
        if word['text'].lower() in self.patterns_dict['conditional_adverbial_subordinators']:
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
        if not self.helper.is_first_word(word):
            prev_word = self.get_previous_word(word_index)
            if self.helper.is_punctuation(prev_word) \
                    and word['text'].lower() in self.patterns_dict['discourse_particles']:
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
        if word['text'].lower() in self.patterns_dict['conjucts']:
            return 'CONJ'

        if not self.is_last_word_in_sentence(word_index):
            prev_word = self.get_previous_word(word_index)
            if self.helper.is_punctuation(prev_word) and word['text'].lower() in ['altogether', 'rather']:
                return 'CONJ'

    def tag_ger(self, word, word_index):
        """ Gerunds with length > 10 are nominal form (N) that ends in –ing or –ings """
        if len(word['text']) > 10 and "VerbForm=Ger" in word['feats']:
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

            # Handle the 'soon as' case
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
        if word_index > 0 and not self.is_last_word_in_sentence(word_index):
            prev_word = self.get_previous_word(word_index)
            next_word = self.get_next_word(word_index)
            if self.helper.is_punctuation(prev_word) and word['upos'] == 'VBG':
                if (self.helper.is_adposition(next_word) and next_word['xpos'] == 'IN') or \
                        next_word['xpos'] in ['DT', 'QUAN', 'CD', 'WH', 'WP', 'WP$', 'PRP', 'RB']:
                    return 'PRESP'

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
        if word['xpos'] == 'VBG' and not self.helper.is_first_word(word):
            prev_word = self.get_previous_word(word_index)
            if self.helper.is_noun(prev_word):
                return 'WZPREZ'

    def tag_pass(self, word, word_index):
        """ Agentless passives are tagged for 2 patterns. First, any form BE + (VBD|VBN) with 1-2 optional
        intervening RBs or negations. Second any form BE + nominal form (noun|pronoun) + (VBN|VBD)
        with optional negation. Original Biber did not allow for interverning negation in this pattern"""
        pass

    def tag_jj(self, word, word_index):
        """ Attributive adjectives """
        if word['xpos'] in ['JJ', 'JJR', 'JJS']:
            return 'JJ'

    def tag_bema(self, word, word_index):
        """ Be as main verb (BEMA): BE followed by a (DT), (PRP$) or a (PIN) or an adjective (JJ). Modified from
        the original algorithm to take adverbs and negations into account. Also no double coding
        with existential there """
        pass

    def tag_osub(self, word, word_index):
        """ Other adverbial subordinators. Any occurrence of the OSUB words. For multi-word units only tag the first """
        if word['text'].lower() in self.patterns_dict['other_adverbial_subordinators']:
            return "OSUB"

        # TODO: Come back and finish this. Multi-words are annoying

    def tag_phc(self, word, word_index):
        """ Phrasal coordination. Any 'and' followed by the same tag if the tag is in (adverb, adjective, verb, noun)"""
        if word['text'].lower() == 'and':
            if not self.helper.is_first_word(word) and not self.is_last_word_in_sentence(word_index):
                prev_word = self.get_previous_word(word_index)
                next_word = self.get_next_word(word_index)
                if prev_word['upos'] == next_word['upos']:
                    if self.helper.is_adverb(prev_word) or self.helper.is_adjective(prev_word) or \
                            self.helper.is_verb(prev_word) or self.helper.is_noun(
                        prev_word) or self.helper.is_proper_noun(prev_word):
                        return "PHC"

    def tag_prmd(self, word, word_index):
        """ Predictive modals. will, would, shall and their contractions: ‘d_MD, ll_MD, wo_MD, sha_MD"""
        # TODO: Revisit contractions
        if word['text'].lower() in self.patterns_dict['predictive_modals']:
            return 'PRMD'

    def tag_sere(self, word, word_index):
        """ Sentence relatives. Everytime a punctuation mark is followed by the word which """
        if not self.is_last_word_in_sentence(word_index):
            next_word = self.get_next_word(word_index)
            if self.helper.is_punctuation(word) and next_word['text'].lower() == 'which':
                return 'SERE'

    def tag_stpr(self, word, word_index):
        """ Stranded preposition. Preposition followed by a punctuation mark.
        Update from Biber: can't be the word besides. E.g. the candidates I was thinking 'of',"""
        if not self.is_last_word_in_sentence(word_index):
            next_word = self.get_next_word(word_index)
            if self.helper.is_adposition(word) and word['xpos'] == 'IN' and self.helper.is_punctuation(next_word):
                return 'STPR'

    def tag_pin(self, word, word_index):
        """ Total prepositional phrases. Preposition 'to' is disambiguated from the infinitive marker to.
        Biber doesn't mention if he differentiates between the 2 """
        if self.helper.is_adposition(word) and word['xpos'] == 'IN':
            return 'PIN'

    def tag_nomz(self, word, word_index):
        """ Any noun ending in -tion, -ment, -ness. (TODO: Add -ity?)"""
        if self.helper.is_noun(word) and (word['text'][-4:].lower() in ['tion', 'ment', 'ness'] or
                                          word['text'][-5:].lower() in ['tions', 'ments'] or
                                          word['text'][-6:].lower() == 'nesses'):
            return 'NOMZ'
