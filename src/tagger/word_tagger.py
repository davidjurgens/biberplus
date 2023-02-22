import inspect

import numpy as np

from src.tagger.run_order import RUN_ORDER
from src.tagger.tag_helper import TagHelper


class WordTagger:
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

        # Get all the methods that start with 'tag' and sort them by run order
        methods = filter(inspect.ismethod, attrs)
        tag_methods = [m for m in methods if m.__name__[:3] == 'tag']
        tag_methods = sorted(tag_methods, key=lambda x: RUN_ORDER.index(x.__name__))

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

    def tag_pire(self, word, word_index):
        """ Pied-piping relatives clauses """

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

    def tag_tsub(self, word, word_index):
        """ That relative clauses on subject position: that  preceded by a noun (N) and followed by an
        auxiliary verb or a verb (V), with the possibility of an intervening adverb (RB) or negation (XX0)
        e.g. the dog 'that bit me' """
        pass

    def tag_demp(self, word, word_index):
        """ The program tags as demonstrative pronouns the words those, this, these when they are
            followed by a verb (any tag starting with V) or auxiliary verb (modal verbs in the form of
            MD tags or forms of DO or forms of HAVE or forms of BE) or a punctuation mark or a WH
            pronoun or the word and. The word that is tagged as a demonstrative pronoun when it
            follows the said pattern or when it is followed by ‘s or is and, at the same time, it has not
            been already tagged as a TOBJ, TSUB, THAC or THVC.
        """
        pass

    def tag_whcl(self, word, word_index):
        """ WH-clauses. any public, private or suasive verb followed by any WH word, followed by a word that is
        NOT an auxiliary (tag MD for modal verbs, or a form of DO, or a form of HAVE, or a form of BE)."""
        pass

    def tag_whsub(self, word, word_index):
        """ WH relative clauses on subject position. Any word that is NOT a form of the
        words ASK or TELL followed by a noun (N), then a WH pronoun, then by any verb or auxiliary
        verb (V), with the possibility of an intervening adverb (RB) or negation (XX0) between the
        WH pronoun and the verb. e.g. the man who likes popcorn"""
        pass

    def tag_whobj(self, word, word_index):
        """ WH relative clauses on object position. Any word that is NOT a form of the words ASK or TELL followed by any word, followed by a
        noun (N), followed by any word that is NOT an adverb (RB), a negation (XX0) , a verb or an auxiliary
        verb (MD, forms of HAVE, BE or DO) e.g. the man who Sally likes """
        pass

    def tag_nn(self, word, word_index):
        """ Total other nouns. Any noun not tagged as a nominalisation or a gerund. Plural nouns (NNS) and
        proper nouns (NNP and NNPS) tags are changed to NN and included in this count"""
        pass

    def tag_thvc(self, word, word_index):
        """ That verb complements. Tag is assigned when the word that is: (1) preceded by and, nor, but, or,
        also or any punctuation mark and followed by a determiner (DT, QUAN, CD), a pronoun (PRP), there, a plural
        noun (NNS) or a proper noun (NNP); (2) preceded by a public, private or suasive verb or a form of seem or
        appear and followed by any word that is NOT a verb (V), auxiliary verb (MD, form of DO, form of HAVE,
        form of BE), a punctuation or the word and; (3) preceded by a public, private or suasive verb or a form
        of seem or appear and a preposition and up to four words that are not nouns (N) """
        pass

    def tag_tobj(self, word, word_index):
        """ That relative clauses on object position. These are occurrences of that preceded by a noun and
        followed by a determiner (DT, QUAN, CD), a subject form of a personal pronoun, a possessive pronoun
        (PRP$), the pronoun it, an adjective (JJ), a plural noun (NNS), a proper noun (NNP) or a possessive
        noun (a noun (N) followed by a genitive marker (POS)). e.g. the dog that I saw"""
        pass

    def tag_thatd(self, word, word_index):
        """ Subordinator that deletion. A public, private or suasive verb followed by a demonstrative
        pronoun (DEMP) or a subject form of a personal pronoun; 2) a public, private or suasive verb is
        followed by a pronoun (PRP) or a noun (N) and then by a verb (V) or auxiliary verb; (3) a
        public, private or suasive verb is followed by an adjective (JJ or PRED), an adverb (RB),
        a determiner (DT, QUAN, CD) or a possessive pronoun (PRP$) and then a noun (N) and then a verb or
        auxiliary verb, with the possibility of an intervening adjective (JJ or PRED) between the noun and
        ts preceding word """
        pass

    def tag_spin(self, word, word_index):
        """ Split infinitives. every time an infinitive marker to is followed by one or two adverbs and
        a verb base form. e.g. he wants to convincingly prove that """
        pass

    def tag_spau(self, word, word_index):
        """ Split auxiliaries. Auxiliary (any modal verb MD, or any form of DO, or any form of BE, or any
        form of HAVE) is followed by one or two adverbs and a verb base form"""
        pass

    def tag_prod(self, word, word_index):
        """ Pro-verb do. Any form of DO that is used as main verb and, therefore, excluding DO when used as
        auxiliary verb. The tagger tags as PROD any DO that is NOT in neither of the following patterns:
        (a) DO followed by a verb (any tag starting with V) or followed by adverbs (RB), negations and then
        a verb (V); (b) DO preceded by a punctuation mark or a WH pronoun (the list of WH pronouns
        is in Biber (1988))"""
        pass


    def tag_pred(self, word, word_index):
        """ Predicative adjectives. Any form of BE followed by an adjective (JJ) followed by a word that is NOT
        another adjective, an adverb (RB) or a noun (N). If any adverb or negation is intervening between the
        adjective and the word after it, the tag is still assigned. An adjective is tagged as predicative if it is
        preceded by another predicative adjective followed by a phrasal coordinator e.g. the horse is big and fast """
        pass

    def tag_peas(self, word, word_index):
        """ Perfect aspect. Calculated by counting how many times a form of HAVE is followed by: a VBD or VBN tag
        (a past or participle form of any verb). These are also counted when an adverb (RB) or negation (XX0)
        occurs between the two. The interrogative version is counted too. This is achieved by counting how many
        times a form of HAVE is followed by a nominal form (noun, NN, proper noun, NP or personal pronoun, PRP)
        and then followed by a VBD or VBN tag. As for the affirmative version, the latter algorithm also accounts
        for intervening adverbs or negations. """
        pass


    def tag_andc(self, word, word_index):
        """ Independent clause coordination. Assigned to the word and when it is found in one of the following
         patterns: (1) preceded by a comma and followed by it, so, then, you, there + BE, or a demonstrative pronoun
         (DEMP) or the subject forms of a personal pronouns; (2) preceded by any punctuation; (3) followed by a
         WH pronoun or any WH word, an adverbial subordinator (CAUS, CONC, COND, OSUB) or a discourse particle (DPAR)
         or a conjunct (CONJ)"""
        pass

    def tag_hdg(self, word, word_index):
        """ Hedges. Any hedge token. In cases of multi-word units such as more or less, only the first word is
        tagged as HDG. For the terms sort of and kind of hese two items must be preceded by a determiner (DT),
        a quantifier (QUAN), a cardinal number (CD), an adjective (JJ or PRED), a possessive pronouns (PRP$) or
        WH word (see entry on WH-questions """
        pass

    def tag_emph(self, word, word_index):
        """ Emphatics. Any word in the emphatics list and real+adjective, so+adjective, any form of DO followed
        by a verb, for sure, a lot, such a. In cases of multi- word units such as a lot,
        only the first word is tagged """
        pass

    def tag_whqu(self, word, word_index):
        """ Direct WH-questions. Punctuation + WH word + auxillary verb. Slightly modified to allow
        for intervening word between punctuation and WH word"""
        pass

    def tag_demo(self, word, word_index):
        """ Demonstratives. words that, this, these, those have not been
        tagged as either DEMP, TOBJ, TSUB, THAC, or THVC"""
        pass

    def tag_bypa(self, word, word_index):
        """ By-passives. PASS are found and the preposition by follows it"""
        pass

    def tag_thac(self, word, word_index):
        """ That adjective complements """
        pass