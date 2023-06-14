import inspect
import re

import numpy as np

from bibermda.tagger.biber_run_order import RUN_ORDER
from bibermda.tagger.tag_helper import TagHelper


class BiberPlusTagger:
    def __init__(self, tagged_words, patterns_dict, ttr_n=400):
        """
            :param tagged_words: Words in the form of a dictionary
            :param patterns_dict: Dictionary containing list of words for different patterns.
            e.g. public verbs, downtoners, etc.
        """
        self.tagged_words = tagged_words
        self.word_count = len(self.tagged_words)
        self.patterns = patterns_dict
        self.helper = TagHelper(patterns_dict)
        self.word_lengths = []
        self.adverb_count = 0
        self.mean_word_length = -1
        self.current_index = 0

        if ttr_n > self.word_count:
            self.ttr_n = self.word_count
        else:
            self.ttr_n = ttr_n

        self.ttr = -1.0

    def run_all(self):
        """ Run all tagger functions defined in this class"""
        attrs = (getattr(self, name) for name in dir(self))

        # Get all the methods that start with 'tag' and sort them by run order
        methods = filter(inspect.ismethod, attrs)
        tag_methods = [m for m in methods if m.__name__[:3] == 'tag']
        tag_methods = sorted(tag_methods, key=lambda x: RUN_ORDER.index(x.__name__))

        for index, tagged_word in enumerate(self.tagged_words):
            # Get the context surrounding the word. Sets to None if unavailable
            previous_words = self.get_previous_n_words(index, n=4)
            next_words = self.get_next_n_words(index, n=4)
            self.current_index = index

            for tag_method in tag_methods:
                tag = tag_method(tagged_word, previous_words, next_words)
                if tag:
                    tagged_word['tags'].append(tag)

            self.update_doc_level_stats(tagged_word)
        self.mean_word_length = np.array(self.word_lengths).mean()

        return self.tagged_words

    """ Helper functions """

    def update_doc_level_stats(self, word):
        if self.helper.is_adverb(word):
            self.adverb_count += 1

        self.word_lengths.append(len(word['text']))

    def get_previous_n_words(self, index, n):
        if index - n < 0:
            padding = [None] * (n - index)
            n -= len(padding)
            previous_n_words = self.tagged_words[index - n: n][::-1]
            previous_n_words.extend(padding)
            return previous_n_words
        else:
            return self.tagged_words[index - n: index][::-1]

    def get_next_n_words(self, index, n):
        next_n_words = self.tagged_words[index + 1: index + n + 1]
        if len(next_n_words) < n:
            padding = [None] * (n - len(next_n_words))
            next_n_words.extend(padding)
        return next_n_words

    def get_phrase(self, next_n_words):
        curr_word_txt = self.tagged_words[self.current_index]['text'].lower()
        next_n_text = [w['text'].lower() for w in next_n_words]
        phrase = curr_word_txt + " " + " ".join(next_n_text)
        return phrase

    """ A) Tense and Aspect Markers """

    def tag_vbd(self, word, previous_words, next_words):
        """ Past tense POS """
        if word['xpos'] == 'VBD':
            return 'VBD'

    def tag_peas(self, word, previous_words, next_words):
        """ Perfect aspect
         1) HAVE + (ADV) + (ADV) + VBD/VBN
         2) HAVE + N/PRO + VBN/VBD
         """
        # HAVE + VBD/VBN
        if self.helper.is_have(word):
            if next_words[0] and next_words[0]['xpos'] in ['VBD', 'VBN']:
                return "PEAS"

        if next_words[1]:
            if next_words[1]['xpos'] in ['VBD', 'VBN']:
                if self.helper.is_adverb(next_words[0]) or self.helper.is_noun(next_words[0]) or \
                        self.helper.is_pronoun(next_words[0]):
                    return "PEAS"

        # HAVE + (ADV) + (ADV) + VBD/VBN
        if next_words[2]:
            if self.helper.is_adverb(next_words[0]) and self.helper.is_adverb(next_words[1]) and \
                    next_words[2]['xpos'] in ['VBD', 'VBN']:
                return "PEAS"

    def tag_vprt(self, word, previous_words, next_words):
        """ Present tense: VBP or VBZ tag"""
        if word['xpos'] in ['VBP', 'VBZ']:
            return 'VPRT'

    """ B) PLACE and TIME Adverbials """

    def tag_place(self, word, previous_words, next_words):
        """ Any item in the place adverbials list that is not a proper noun (NNP) """
        if word['text'].lower() in self.patterns['place_adverbials'] and not self.helper.is_proper_noun(word):
            return 'PLACE'

    def tag_time(self, word, previous_words, next_words):
        """ Time adverbials with the exception: soon is not a time adverbial if it is followed by the word as """
        if word['text'].lower() in self.patterns['time_adverbials']:
            return 'TIME'

        # Handle the 'soon as' case
        if word['text'].lower() == 'soon':
            if next_words[0] and next_words[0]['text'].lower() != 'as':
                return 'TIME'

    """ C) Pronouns and pro-verbs """

    def tag_fpp1(self, word, previous_words, next_words):
        """ Any item of this list: I, me, us, my, we, our, myself, ourselves and their contractions.
        Tokenizer separates contractionas """
        if word['text'].lower() in self.patterns['first_person_pronouns']:
            return 'FPP1'

    def tag_spp2(self, word, previous_words, next_words):
        """ Any item of this list: you, your, yourself, yourselves, thy, thee, thyself, thou """
        if word['text'].lower() in self.patterns['second_person_pronouns']:
            return 'SPP2'

    def tag_tpp3(self, word, previous_words, next_words):
        """ Any item of this list: she, he, they, her, him, them, his, their, himself, herself, themselves """
        if word['text'].lower() in self.patterns['third_person_pronouns']:
            return 'TPP3'

    def tag_pit(self, word, previous_words, next_words):
        """ Any pronoun it. Although not specified in Biber (1988), the present program also tags its and
            itself as “Pronoun it”. """
        if self.helper.is_pronoun(word) and word['text'].lower() in self.patterns['pronoun_it']:
            return 'PIT'

    def tag_inpr(self, word, previous_words, next_words):
        """ Any item of this list: anybody, anyone, anything, everybody, everyone, everything, nobody,
            none, nothing, nowhere, somebody, someone, something """
        if word['text'].lower() in self.patterns['indefinite_pronouns']:
            return 'INPR'

    def tag_demp(self, word, previous_words, next_words):
        """ The program tags as demonstrative pronouns the words those, this, these when they are
            followed by a verb (any tag starting with V) or auxiliary verb (modal verbs in the form of
            MD tags or forms of DO or forms of HAVE or forms of BE) or a punctuation mark or a WH
            pronoun or the word and. The word that is tagged as a demonstrative pronoun when it
            follows the said pattern or when it is followed by ‘s or is and, at the same time, it has not
            been already tagged as a TOBJ, TSUB, THAC or THVC.
        """
        if word['text'].lower() in self.patterns['demonstrative_pronouns']:
            if not ('TOBJ' in word['tags'] or 'TSUB' in word['tags'] or 'THAC' in word['tags']
                    or 'THVC' in word['tags']):

                if next_words[0] and (self.helper.is_any_verb(next_words[0])
                                      or next_words[0]['text'].lower() in ["'s", "and"] or
                                      self.helper.is_punctuation(next_words[0])
                                      or next_words[0]['xpos'] == 'WP'):
                    return "DEMP"

    def tag_prod(self, word, previous_words, next_words):
        """ Pro-verb do. Any form of DO that is used as main verb and, therefore, excluding DO when used as
        auxiliary verb. The tagger tags as PROD any DO that is NOT in neither of the following patterns:
        (a) DO followed by a verb (any tag starting with V) or followed by adverbs (RB), negations and then
        a verb (V); (b) DO preceded by a punctuation mark or a WH pronoun (the list of WH pronouns
        is in Biber (1988))"""
        if self.helper.is_do(word):
            # Exclude DO + Verb
            if next_words[0] and self.helper.is_verb(next_words[0]):
                return

            # Exclude DO + Adverb + Verb
            if next_words[1] and self.helper.is_adverb(next_words[0]) and self.helper.is_verb(next_words[1]):
                return

            # Exclude PUNCT + DO and WHP + DO
            if previous_words[0] and (self.helper.is_punctuation(previous_words[0]) or
                                      previous_words[0]['xpos'] == 'WP'):
                return

            return "PROD"

    """ D) Questions """

    def tag_whqu(self, word, previous_words, next_words):
        """ Direct WH-questions. Punctuation + WH word + auxillary verb. Slightly modified to allow
        for intervening word between punctuation and WH word"""
        if word['xpos'][0] == 'W':
            if previous_words[0] and next_words[0] and self.helper.is_auxiliary(next_words[0]):
                if self.helper.is_punctuation(previous_words[0]):
                    return "WHQU"

                # Cause of intervening word between punctuation and WH word
                if previous_words[1] and self.helper.is_punctuation(previous_words[1]):
                    return "WHQU"

    """ E) Nominal Forms """

    def tag_nomz(self, word, previous_words, next_words):
        """ Any noun ending in -tion, -ment, -ness """
        if self.helper.is_noun(word) and (word['text'][-3:].lower() == 'ity' or
                                          word['text'][-4:].lower() in ['tion', 'ment', 'ness'] or
                                          word['text'][-5:].lower() in ['tions', 'ments'] or
                                          word['text'][-6:].lower() == 'nesses'):
            if word['text'].lower() not in self.patterns['nominalizations_stop_list']:
                return 'NOMZ'

    def tag_ger(self, word, previous_words, next_words):
        """ Gerunds with length >= 10 are nominal form (N) that ends in –ing or –ings """
        if len(word['text']) >= 10 and word['xpos'][0] == 'N' and (
                word['text'][-3:].lower() == 'ing' or word['text'][-4:].lower() == 'ings'):
            return 'GER'

    def tag_nn(self, word, previous_words, next_words):
        """ Total other nouns. Any noun not tagged as a nominalisation or a gerund. Plural nouns (NNS) and
        proper nouns (NNP and NNPS) tags are changed to NN and included in this count"""
        if self.helper.is_noun(word) and 'NOMZ' not in word['tags'] and 'GER' not in word['tags']:
            return "NN"

    """ F) Passives """

    def tag_pass(self, word, previous_words, next_words):
        """ Agentless passives are tagged for 2 patterns. First, any form BE + 1-2 optional RBs + (VBD|VBN).
        Second any form BE + nominal form (noun|pronoun) + (VBN). Following original Biber which does not allow
        for intervening negation in this pattern"""
        if self.helper.is_be(word):
            # Case of BE + VBN/VBD
            if next_words[0] and next_words[0]['xpos'] in ['VBN', 'VBD']:
                return "PASS"

            if next_words[1]:
                # Case of BE + (ADV) + VBN/VBD
                if self.helper.is_adverb(next_words[0]) and next_words[1]['xpos'] in ['VBN', 'VBD']:
                    return "PASS"

                # Case of BE + N/PRO + VBN/VBD
                if (self.helper.is_noun(next_words[0]) or self.helper.is_pronoun(next_words[0])) and \
                        next_words[1]['xpos'] in ['VBN', 'VBD']:
                    return "PASS"

            # Case of BE + (ADV) + (ADV) + VBN/VBD
            if next_words[2] and self.helper.is_adverb(next_words[0]) and self.helper.is_adverb(next_words[1]) and \
                    next_words[2]['xpos'] in ['VBN', 'VBD']:
                return "PASS"

    def tag_bypa(self, word, previous_words, next_words):
        """ By-passives. PASS are found and the preposition by follows it"""
        if 'PASS' in word['tags']:
            for i in range(min(len(next_words), 4)):
                if next_words[i] and next_words[i]['text'].lower() == 'by':
                    return 'BYPA'

    """ G) Stative Forms"""

    def tag_bema(self, word, previous_words, next_words):
        """ Be as main verb (BEMA): BE followed by a (DT), (PRP$) or a (PIN) or an adjective (JJ). Slight modification
        from Biber. Allow adverbs or negations to appear between the verb BE and the rest of the pattern """
        if not self.helper.is_be(word):
            return None

            # Check the first possible pattern: BE + (DT|PRP$|JJ|JJR|PIN)
        if next_words[0] and (
                next_words[0]['xpos'] in ['DT', 'PRP$', 'JJ', 'JJR'] or self.helper.is_preposition(next_words[0])):
            return 'BEMA'

            # Check the second possible pattern: BE + (Adverb|Negation) + (DT|PRP$|JJ|JJR|PIN)
        if next_words[1] and (self.helper.is_adverb(next_words[0]) or self.tag_xx0(next_words[0], None, None)) and (
                next_words[1]['xpos'] in ['DT', 'PRP$', 'JJ', 'JJR'] or self.helper.is_preposition(next_words[1])):
            return 'BEMA'

    def tag_ex(self, word, previous_words, next_words):
        """ Existential there from the POS tags"""
        if word['xpos'] == 'EX':
            return 'EX'

    """ H) Subordination Features """

    def tag_thvc(self, word, previous_words, next_words):
        """ That verb complements. Tag is assigned when the word that is:
        (1) preceded by and, nor, but, or, also or any punctuation mark and followed by a determiner (DT, QUAN, CD),
        a pronoun (PRP), there, a plural noun (NNS) or a proper noun (NNP);
        (2) preceded by a public, private or suasive verb or a form of seem or appear and followed by any word that
        is NOT a verb (V), auxiliary verb (MD, form of DO, form of HAVE, form of BE), a punctuation or the word and
        (3) preceded by a public, private or suasive verb or a form of seem or appear and a preposition
        and up to four words that are not nouns (N) """

        def is_rel_verb(curr_word):
            txt = curr_word['text'].lower()
            return txt in self.patterns['public_verbs'] or txt in self.patterns['private_verbs'] or \
                   txt in self.patterns['suasive_verbs']

        if word['text'].lower() == 'that':
            if previous_words[0] and next_words[0]:
                # Handle first case
                if previous_words[0]['text'].lower() in ['and', 'nor', 'but', 'or', 'also'] \
                        or self.helper.is_punctuation(previous_words[0]):
                    if next_words[0]['xpos'] in ['DT', 'CD', 'PRP', 'NNS', 'NNP'] \
                            or self.helper.is_quantifier(next_words[0]):
                        return "THVC"
                # Handle second case
                if is_rel_verb(previous_words[0]) or 'SMP' in previous_words[0]['tags']:
                    if not (self.helper.is_any_verb(next_words[0]) or self.helper.is_punctuation(next_words[0]) or
                            next_words[0]['text'].lower() == 'and'):
                        return "THVC"
                # Handle third case
                # PUB/PRIV/SUA + PREP + xxxx + N + that

                if self.helper.is_noun(previous_words[0]):
                    # PUB/PRIV/SUA + PREP + N + that
                    if previous_words[2] and is_rel_verb(previous_words[2]) and self.helper.is_preposition(
                            previous_words[1]):
                        return "THVC"

                    # Allow for up to 4 intervening words that are not nouns
                    for i in range(1, 5):

                        prev_n_words = self.get_previous_n_words(self.current_index, n=i + 3)
                        if prev_n_words[i + 2] and is_rel_verb(prev_n_words[0]) and self.helper.is_preposition(
                                prev_n_words[1]):
                            if not any(self.helper.is_noun(w) for w in prev_n_words[2:i + 1]):
                                return "THVC"

    def tag_thac(self, word, previous_words, next_words):
        """ That adjective complements. That preceded by an adjective (JJ or a predicative adjective, PRED)."""
        if word['text'].lower() == 'that' and previous_words[0]:
            if self.helper.is_adjective(previous_words[0]) or 'PRED' in previous_words[0]['tags']:
                return "THAC"

    def tag_whcl(self, word, previous_words, next_words):
        """ WH-clauses. any public, private or suasive verb followed by any WH word, followed by a word that is
        NOT an auxiliary (tag MD for modal verbs, or a form of DO, or a form of HAVE, or a form of BE)."""

        if word['xpos'][0] == 'W' and previous_words[0]:
            if 'PUBV' in previous_words[0]['tags'] or 'PRIV' in previous_words[0]['tags'] or 'SUAV' in \
                    previous_words[0]['tags']:
                if next_words[0] and not self.helper.is_auxiliary(next_words[0]):
                    return "WHCL"

    def tag_to(self, word, previous_words, next_words):
        """ Infinitives: POS tag TO that are not a preposition """
        if word['xpos'] == 'TO':
            # Prepositions are 'to's followed by
            filter_preps = ['IN', 'CD', 'DT', 'JJ', 'PRP$', 'WP$', 'WDT', 'WP', 'WRB',
                            'PDT', 'N', 'NNS', 'NP', 'NPs', 'PRP']
            if next_words[0] and next_words[0]['xpos'] not in filter_preps:
                return 'TO'

    def tag_presp(self, word, previous_words, next_words):
        """ Present participial clauses: a punctuation mark is followed by a present participial
        form of a verb (VBG) followed by a preposition (PIN), a determiner (DT, QUAN, CD), a WH pronoun,
        a WH possessive pronoun (WP$), any WH word, any pronoun (PRP) or any adverb (RB)-"""
        if word['xpos'] == 'VBG':
            if previous_words[0] and next_words[0] and self.helper.is_punctuation(previous_words[0]):
                if self.helper.is_preposition(next_words[0]) or next_words[0]['xpos'] in \
                        ['DT', 'QUAN', 'CD', 'WH', 'WP', 'WP$', 'PRP', 'RB', 'WRB']:
                    return 'PRESP'

    def tag_pastp(self, word, previous_words, next_words):
        """ Past partcipial clauses: punctuation followed by VBN -> PIN or RB
         e.g. 'Built' in a single week, the house would stand for fifty years"""
        if word['xpos'] == 'VBN':
            if previous_words[0] and next_words[0]:
                if self.helper.is_punctuation(previous_words[0]) and (self.helper.is_adverb(next_words[0]) or
                                                                      self.tag_pin(next_words[0], None, None)):
                    return "PASTP"

    def tag_wzpast(self, word, previous_words, next_words):
        """ Past participial WHIZ deletion relatives: a noun (N) or quantifier pronoun (QUPR) followed by a
        past participial form of a verb (VBN) followed by a preposition (PIN) or an adverb (RB) or a form of BE.
        e.g. The solution 'produced' by this process """
        if word['xpos'] == 'VBN' and previous_words[0]:
            if self.helper.is_noun(previous_words[0]) or self.helper.is_quantifier_pronoun(previous_words[0]):
                if next_words[0] and (self.helper.is_preposition(next_words[0]) or self.helper.is_adverb(
                        next_words[0]) or self.helper.is_be(next_words[0])):
                    return "WZPAST"

    def tag_wzpres(self, word, previous_words, next_words):
        """ Present participial WHIZ deletion relatives: VBG preceded by an NN
        e.g. the 'causing' this decline' is """
        if word['xpos'] == 'VBG':
            if previous_words[0] and previous_words[0]['xpos'][:2] == 'NN':
                return 'WZPRES'

    def tag_tsub(self, word, previous_words, next_words):
        """ That relative clauses on subject position: that preceded by a noun (N) and followed by an
        auxiliary verb or a verb (V), with the possibility of an intervening adverb (RB) or negation (XX0)
        e.g. the dog 'that bit me' """
        if word['text'].lower() == 'that':
            if previous_words[0] and next_words[0]:
                if self.helper.is_noun(previous_words[0]) and (self.helper.is_verb(next_words[0]) or
                                                               self.helper.is_auxiliary(next_words[0])):
                    return "TSUB"

            # Allow for intervening RB or XXO
            if next_words[1] and self.tag_xx0(next_words[0], None, None) and self.helper.is_any_verb(next_words[1]):
                return "TSUB"

    def tag_tobj(self, word, previous_words, next_words):
        """ That relative clauses on object position. These are occurrences of that preceded by a noun and
        followed by a determiner (DT, QUAN, CD), a subject form of a personal pronoun, a possessive pronoun
        (PRP$), the pronoun it, an adjective (JJ), a plural noun (NNS), a proper noun (NNP) or a possessive
        noun (a noun (N) followed by a genitive marker (POS)). e.g. the dog that I saw"""
        if word['text'].lower() == 'that':
            if previous_words[0] and next_words[0] and self.helper.is_noun(previous_words[0]):
                # N + that + DT|QUAN|CD|PerPro|PRP$|PIT|JJ|NNS|NNP|PossNoun
                if next_words[0]['xpos'] in ['DT', 'CD', 'PRP', 'PRP$', 'NNS', 'NNP', 'JJ'] or \
                        self.helper.is_quantifier(next_words[0]) or self.tag_pit(next_words[0], previous_words[1:],
                                                                                 next_words[1:]) \
                        or self.helper.is_possesive_pronoun(next_words[0]):
                    return "TOBJ"

    def tag_whsub(self, word, previous_words, next_words):
        """ WH relative clauses on subject position. Any word that is NOT a form of the
        words ASK or TELL followed by a noun (N), then a WH pronoun, then by any verb or auxiliary
        verb (V), with the possibility of an intervening adverb (RB) or negation (XX0) between the
        WH pronoun and the verb. e.g. the man who likes popcorn"""
        # TODO: Revist ASK or TELL form
        if word['xpos'][0] == 'W':
            if previous_words[1] and previous_words[1]['text'].lower() not in self.patterns['ask_tell'] \
                    and self.helper.is_noun(previous_words[0]):
                # NOT ASK/TELL -> Noun -> WP -> Verb
                if next_words[0] and self.helper.is_any_verb(next_words[0]):
                    return "WHSUB"

                # NOT ASK/TELL -> Noun -> WP -> RB/XXO -> Verb
                if next_words[1] and (self.helper.is_adverb(next_words[0]) or
                                      self.tag_xx0(next_words[0], None, None)) \
                        and self.helper.is_any_verb(next_words[1]):
                    return "WHSUB"

    def tag_whobj(self, word, previous_words, next_words):
        """ WH relative clauses on object position. Any word that is NOT a form of the words ASK or TELL followed by
        any word, followed by a noun (N), followed by any word that is NOT an adverb (RB), a negation (XX0),
        a verb or an auxiliary verb (MD, forms of HAVE, BE or DO)
        e.g. the man who Sally likes """
        if word['xpos'][0] == 'W':
            # xxx + yyy + N + WP + zzz
            if next_words[0] and not (self.helper.is_adverb(next_words[0]) or self.helper.is_any_verb(next_words[0])
                                      or self.tag_xx0(next_words[0], previous_words[1:], next_words[1:])):
                if previous_words[2] and previous_words[2]['text'].lower() not in self.patterns['ask_tell'] \
                        and self.helper.is_noun(previous_words[0]):
                    return "WHOBJ"

    def tag_pire(self, word, previous_words, next_words):
        """ Pied-piping relatives clauses. Any preposition (PIN) followed by whom, who, whose or which """
        if word['text'].lower() in ['whom', 'who', 'whose', 'which']:
            if previous_words[0] and 'PIN' in previous_words[0]['tags']:
                return "PIRE"

    def tag_sere(self, word, previous_words, next_words):
        """ Sentence relatives. Everytime a punctuation mark is followed by the word which """
        if word['text'].lower() == 'which':
            if previous_words[0] and self.helper.is_punctuation(previous_words[0]):
                return 'SERE'

    def tag_caus(self, word, previous_words, next_words):
        """ Any occurrence of the word because """
        if word['text'].lower() == 'because':
            return 'CAUS'

    def tag_conc(self, word, previous_words, next_words):
        """ Any occurrence of the words although, though, tho """
        if word['text'].lower() in self.patterns['concessive_adverbial_subordinators']:
            return 'CONC'

    def tag_cond(self, word, previous_words, next_words):
        """ Any occurrence of the words if or unless"""
        if word['text'].lower() in self.patterns['conditional_adverbial_subordinators']:
            return 'COND'

    def tag_osub(self, word, previous_words, next_words):
        """ Other adverbial subordinators. Any occurrence of the OSUB words. For multi-word units only tag the first """
        # One word case
        if word['text'].lower() in self.patterns['other_adverbial_subordinators']:
            return "OSUB"

        # 2 word case
        if next_words[0]:
            phrase = self.get_phrase(next_words[:1])
            if phrase in self.patterns['other_adverbial_subordinators']:
                return "OSUB"

        # 3 word case
        if next_words[1]:
            phrase = self.get_phrase(next_words[:2])
            if phrase in self.patterns['other_adverbial_subordinators']:
                return "OSUB"

            # So that and such that cases
            if phrase in ['so that', 'such that']:
                # Cannot be followed by a noun or adjective
                if not (self.helper.is_noun(next_words[1]) or self.helper.is_adjective(next_words[1])):
                    return "OSUB"

    """ I) Prepositional Phrases, Adjectives, and Adverbs"""

    def tag_pin(self, word, previous_words, next_words):
        """ Total prepositional phrases """
        if self.helper.is_preposition(word):
            return "PIN"

    def tag_jj(self, word, previous_words, next_words):
        """ Attributive adjectives """
        if word['xpos'] in ['JJ', 'JJR', 'JJS']:
            return 'JJ'

    def tag_pred(self, word, previous_words, next_words):
        """ Predicative adjectives. Any form of BE followed by an adjective (JJ) followed by a word that is NOT
        another adjective, an adverb (RB) or a noun (N). If any adverb or negation is intervening between the
        adjective and the word after it, the tag is still assigned.
        An adjective is tagged as predicative if it is
        preceded by another predicative adjective followed by a phrasal coordinator e.g. the horse is big and fast """
        if self.helper.is_adjective(word) and previous_words[0] and self.helper.is_be(previous_words[0]):
            # BE -> ADJ -> NOT (JJ|RB|N)
            if next_words[0] and not (self.helper.is_adverb(next_words[0]) or self.helper.is_noun(next_words[0])):
                return "PRED"
            # Allow for intervening negation/adverb
            # BE -> ADJ -> ADV|XXO -> NOT (JJ|RB|N)
            if next_words[1] and (self.tag_xx0(next_words[0], None, None) or self.helper.is_adverb(next_words[0])) \
                    and not (self.helper.is_adjective(next_words[1]) or self.helper.is_noun(next_words[1])
                             or self.helper.is_adverb(next_words[1])):
                return "PRED"

        # Handle phrasal coordinator case
        if self.helper.is_adjective(word):
            new_next_words = [previous_words[0], word, next_words[0], next_words[1]]
            if previous_words[1] and 'PRED' in previous_words[1]['tags'] and 'PHC' in previous_words[0]['tags']:
                return "PRED"

    def tag_rb(self, word, previous_words, next_words):
        """ Any adverb i.e. POS tags RB, RBS, RBR, WRB"""
        if self.helper.is_adverb(word):
            return 'RB'

    """ J) Lexical Specificity """

    def compute_type_token_ratio(self):
        uniq_vocab = set()

        for i in range(self.ttr_n):
            uniq_vocab.add(self.tagged_words[i]['text'].lower())

        self.ttr = len(uniq_vocab) / self.ttr_n

    """ K) Lexical Classes """

    def tag_conj(self, word, previous_words, next_words):
        """ Conjucts finds any item in the conjucts list with preceding punctuation. Only the first word is tagged """
        if word['text'].lower() in self.patterns['conjucts']:
            return 'CONJ'

        if previous_words[0] and self.helper.is_punctuation(previous_words[0]) and word['text'].lower() \
                in ['altogether', 'rather']:
            return 'CONJ'

    def tag_dwnt(self, word, previous_words, next_words):
        """ Any instance of the words in the downtowners list """
        if word['text'].lower() in self.patterns['downtoners']:
            return 'DWNT'

    def tag_amp(self, word, previous_words, next_words):
        """ Any instance of the items in the amplifiers list """
        if word['text'].lower() in self.patterns['amplifiers']:
            return 'AMP'

    def tag_dpar(self, word, previous_words, next_words):
        """ Discourse particle: the words well, now, anyhow, anyways preceded by a punctuation mark """
        if word['text'].lower() in self.patterns['discourse_particles']:
            if previous_words[0] and self.helper.is_punctuation(previous_words[0]):
                return 'DPAR'

    def tag_hdg(self, word, previous_words, next_words):
        """ Hedges. Any hedge token. In cases of multi-word units such as more or less, only the first word is
        tagged as HDG. For the terms sort of and kind of these two items must be preceded by a determiner (DT),
        a quantifier (QUAN), a cardinal number (CD), an adjective (JJ or PRED), a possessive pronouns (PRP$) or
        WH word (see entry on WH-questions) """
        # One word hedges
        if word['text'].lower() in self.patterns['hedges']:
            return "HDG"

        # Two word hedges
        if next_words[0]:
            phrase = self.get_phrase(next_words[:1])
            # Handle kind of / sort of case
            if phrase in ['kind of', 'sort of']:
                if previous_words[0] and (previous_words[0]['xpos'] in ['DT', 'CD', 'PRP$'] or
                                          previous_words[0]['xpos'][0] == 'W' or
                                          self.helper.is_quantifier(word) or self.helper.is_adjective(
                            previous_words[0])):
                    return "HDG"
            elif phrase in self.patterns['hedges']:
                return "HDG"

        # Three word hedges
        if next_words[1]:
            phrase = self.get_phrase(next_words[:2])
            if phrase in self.patterns['hedges']:
                return "HDG"

    def tag_emph(self, word, previous_words, next_words):
        """ Emphatics. Any word in the emphatics list and real+adjective, so+adjective, any form of DO followed
        by a verb, for sure, a lot, such a. In cases of multi- word units such as a lot,
        only the first word is tagged """
        # One word emphatics
        if word['text'].lower() in self.patterns['emphatics']:
            return "EMPH"

        # Two word emphatics
        if next_words[0]:
            phrase = self.get_phrase(next_words[:1])

            if phrase in self.patterns['emphatics']:
                return "EMPH"

            # Handle (real + adjective) and (so + adjective)
            if word['text'].lower() in ['real', 'so'] and self.helper.is_adjective(next_words[0]):
                return "EMPH"

            # DO form + verb
            if self.helper.is_do(word) and self.helper.is_verb(next_words[0]):
                return "EMPH"

    def tag_demo(self, word, previous_words, next_words):
        """ Demonstratives. words that, this, these, those have not been
        tagged as either DEMP, TOBJ, TSUB, THAC, or THVC"""
        if word['text'].lower() in self.patterns['demonstratives']:
            if not ('DEMP' in word['tags'] or 'TOBJ' in word['tags'] or 'TSUB' in word['tags']
                    or 'THAC' in word['tags'] or 'THVC' in word['tags']):
                return "DEMO"

    """ L) Modals """

    def tag_pomd(self, word, previous_words, next_words):
        """ The possibility modals listed by Biber (1988): can, may, might, could """
        if word['text'].lower() in self.patterns['possibility_modals']:
            return 'POMD'

    def tag_nemd(self, word, previous_words, next_words):
        """ The necessity modals listed by Biber (1988): ought, should, must. """
        if word['text'].lower() in self.patterns['necessity_modals']:
            return 'NEMD'

    def tag_prmd(self, word, previous_words, next_words):
        """ Predictive modals. will, would, shall and their contractions: ‘d_MD, ll_MD, wo_MD, sha_MD"""
        if word['text'].lower() in self.patterns['predictive_modals'] and word['xpos'] == 'MD':
            return 'PRMD'

    """ M) Specialized Verb Classes """

    def tag_pubv(self, word, previous_words, next_words):
        """ Any item in the public verbs list """
        if word['text'].lower() in self.patterns['public_verbs']:
            return 'PUBV'

    def tag_priv(self, word, previous_words, next_words):
        """ Any item in the private verbs list """
        if word['text'].lower() in self.patterns['private_verbs']:
            return 'PRIV'

    def tag_suav(self, word, previous_words, next_words):
        """ Any item in the suasive verbs list """
        if word['text'].lower() in self.patterns['suasive_verbs']:
            return 'SUAV'

    def tag_smp(self, word, previous_words, next_words):
        """ Any occurrence of the forms of the two verbs seem and appear """
        if word['text'].lower() in self.patterns['seem_appear']:
            return 'SMP'

    """ N) Reduced forms and dispreferred structures """

    def tag_cont(self, word, previous_words, next_words):
        """ Any instance of apostrophe followed by a tagged word OR any instance of the item n’t """
        if (word['text'][-3:].lower() == "n't"
                or word['text'][0] == "'" and len(word['text']) > 1):
            return 'CONT'

    def tag_thatd(self, word, previous_words, next_words):
        """ Subordinator that deletion.
        1) A public, private or suasive verb followed by (DEMP) or a subject form of a personal pronoun;
        2) a public, private or suasive verb is followed by a pronoun or noun (N) and then by a verb or auxiliary verb;
        3) a public, private or suasive verb is followed by an adjective (JJ or PRED), an adverb (RB),
        a determiner (DT, QUAN, CD) or a possessive pronoun (PRP$) and then a noun (N) and then a verb or
        auxiliary verb, with the possibility of an intervening adjective (JJ or PRED) between the noun and
        its preceding word """
        txt = word['text'].lower()
        if txt in self.patterns['public_verbs'] or txt in self.patterns['private_verbs'] \
                or txt in self.patterns['suasive_verbs']:

            # PUBV|PRIV|SUAV + DEMP|subject PP
            if next_words[0] and (
                    self.tag_demp(next_words[0], previous_words[1:], next_words[1:]) or next_words[0]['text'].lower()
                    in self.patterns['subject_pronouns']):
                return "THATD"

            # PUBV|PRIV|SUAV + PRO|N + V|AUX
            if next_words[1]:
                if self.helper.is_pronoun(next_words[0]) or self.helper.is_noun(next_words[0]):
                    if self.helper.is_any_verb(next_words[1]):
                        return "THATD"

            # PUBV|PRIV|SUAV + JJ|PRED|ADV|DT|QUAN|CD|PRP$ + N + V|AUXV
            # PUBV|PRIV|SUAV + JJ|PRED|ADV|DT|QUAN|CD|PRP$ + N + V|AUXV
            if next_words[2]:
                next_word_2 = next_words[2]
                if next_word_2['xpos'] in ['JJ', 'DT', 'CD', 'PRP$'] or self.helper.is_adverb(next_word_2) or \
                        self.tag_pred(next_words[0], [word] + previous_words, next_words[1:]) or \
                        self.helper.is_quantifier(next_words[0]):
                    if self.helper.is_noun(next_words[1]) and (self.helper.is_verb(next_word_2) or
                                                               self.helper.is_auxiliary(next_word_2)):
                        return "THATD"

            # PUBV|PRIV|SUAV + JJ|PRED|ADV|DT|QUAN|CD|PRP$ + (ADJ) + N + V|AUXV
            if next_words[3]:
                if next_words[0]['xpos'] in ['JJ', 'DT', 'CD', 'PRP$'] or self.helper.is_adverb(next_words[0]) or \
                        self.tag_pred(next_words[0], previous_words[1:], next_words[1:]) or \
                        self.helper.is_quantifier(next_words[0]):
                    if self.helper.is_adjective(next_words[1]) and self.helper.is_noun(next_words[2]) \
                            and self.helper.is_any_verb(next_words[3]):
                        return "THATD"

    def tag_stpr(self, word, previous_words, next_words):
        """ Stranded preposition. Preposition followed by a punctuation mark.
        Update from Biber: can't be the word besides. E.g. the candidates I was thinking 'of',"""
        if word['text'].lower() in self.patterns['prepositional_phrases'] and word['text'].lower() != 'besides':
            if next_words[0] and self.helper.is_punctuation(next_words[0]):
                return "STPR"

    def tag_spin(self, word, previous_words, next_words):
        """ Split infinitives. every time an infinitive marker to is followed by one or two adverbs and
        a verb base form. e.g. he wants to convincingly prove that """
        if 'TO' in word['tags']:
            # TO + 1 adverb + VB
            if next_words[1] and self.helper.is_adverb(next_words[0]) and next_words[1]['xpos'] == 'VB':
                return "SPIN"

            # TO + 2 adverbs + VB
            if next_words[2] and self.helper.is_adverb(next_words[0]) and self.helper.is_adverb(next_words[1]) and \
                    next_words[2]['xpos'] == 'VB':
                return "SPIN"

    def tag_spau(self, word, previous_words, next_words):
        """ Split auxiliaries. Auxiliary (any modal verb MD, or any form of DO, or any form of BE, or any
        form of HAVE) is followed by one or two adverbs and a verb base form"""
        # Note the existing tagger also uses VBN in addition to verb base form
        if self.helper.is_auxiliary(word):
            # AUX + 1 adverb + VB
            if next_words[1] and self.helper.is_adverb(next_words[0]) and next_words[1]['xpos'][:2] == 'VB':
                return "SPAU"

            # AUX + 2 adverbs + VB
            if next_words[2] and self.helper.is_adverb(next_words[0]) and self.helper.is_adverb(next_words[1]) and \
                    next_words[2]['xpos'][:2] == 'VB':
                return "SPAU"

    """ O) Coordination """

    def tag_phc(self, word, previous_words, next_words):
        """ Phrasal coordination. Any 'and' followed by the same tag if the tag is in (adverb, adjective, verb, noun)"""
        if word['text'].lower() == 'and' and previous_words[0] and next_words[0]:
            if previous_words[0]['upos'] == next_words[0]['upos']:
                if self.helper.is_adverb(previous_words[0]) or self.helper.is_adjective(previous_words[0]) or \
                        self.helper.is_verb(previous_words[0]) or \
                        self.helper.is_noun(previous_words[0]) or self.helper.is_proper_noun(previous_words[0]):
                    return "PHC"

    def tag_andc(self, word, previous_words, next_words):
        """ Independent clause coordination. Assigned to the word and when it is found in one of the following
         patterns:
         (1) preceded by a comma and followed by it, so, then, you, there + BE, or a demonstrative pronoun
         (DEMP) or the subject forms of a personal pronouns;
         (2) preceded by any punctuation;
         (3) followed by a WH pronoun or any WH word, an adverbial subordinator (CAUS, CONC, COND, OSUB) or a
         discourse particle (DPAR) or a conjunct (CONJ)"""
        if word['text'].lower() == 'and':
            # Followed by WH word, CAUS, CONC, COND, OSUB, DPAR, CONJ
            if next_words[0] and (next_words[0]['xpos'] == 'W' or
                                  self.tag_caus(next_words[0], previous_words[1:], next_words[1:]) or
                                  self.tag_conc(next_words[0], previous_words[1:], next_words[1:]) or
                                  self.tag_cond(next_words[0], previous_words[1:], next_words[1:]) or
                                  self.tag_osub(next_words[0], previous_words[1:], next_words[1:]) or
                                  self.tag_dpar(next_words[0], previous_words[1:], next_words[1:]) or
                                  self.tag_conj(next_words[0], previous_words[1:], next_words[1:])):
                return "ANDC"

            # Preceded by any punctuation
            if previous_words[0] and self.helper.is_punctuation(previous_words[0]):
                return "ANDC"

            # Preceded by a comma
            if previous_words[0] and previous_words[0]['text'] == ',':
                # Followed by it, so, then, you or DEMP or personal pronouns
                if next_words[0] and (next_words[0]['text'] in ['it', 'so', 'then', 'you'] or
                                      self.tag_demp(next_words[0], previous_words[1:], next_words[1:])
                                      or next_words[0]['text'].lower() in self.patterns['subject_pronouns']):
                    return "ANDC"
                # Followed by there + BE
                if next_words[1] and next_words[0] == 'there' and self.helper.is_be(next_words[1]):
                    return "ANDC"

    """ P) Negation """

    def tag_xx0(self, word, previous_words, next_words):
        """ Analytic negation: word 'not' and to the item n’t_RB"""
        if word['text'].lower() in self.patterns['analytic_negation']:
            return 'XX0'

    def tag_syne(self, word, previous_words, next_words):
        """ Synthetic negation: (no, neither, nor) followed by an adjective, noun, or proper noun"""
        if word['text'].lower() in self.patterns['synthetic_negations']:
            if next_words[0] and (self.helper.is_adjective(next_words[0]) or self.helper.is_noun(next_words[0])
                                  or self.helper.is_proper_noun(next_words[0])):
                return 'SYNE'

    def tag_quan(self, word, previous_words, next_words):
        if self.helper.is_quantifier(word):
            return "QUAN"

    def tag_qupr(self, word, previous_words, next_words):
        if self.helper.is_quantifier_pronoun(word):
            return "QUPR"

    def tag_articles(self, word, previous_words, next_words):
        if self.helper.is_article(word):
            return "ART"

    def tag_auxillary_be(self, word, previous_words, next_words):
        if word['text'].lower() in self.patterns['be'] and self.helper.is_auxiliary(word):
            return "AUXB"

    def tag_capitalizations(self, word, previous_words, next_words):
        if word['text'][0].isupper():
            return "CAP"

    def tag_subordinating_conjunctions(self, word, previous_words, next_words):
        if self.helper.is_subordinating_conjunction(word):
            return "SCONJ"

    def tag_coordinating_conjunctions(self, word, previous_words, next_words):
        if self.helper.is_coordinating_conjunction(word):
            return 'CCONJ'

    def tag_determiners(self, word, previous_words, next_words):
        if self.helper.is_determiner(word):
            return 'DET'

    def tag_emoji(self, word, previous_words, next_words):
        emoji_pattern = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
        if re.search(emoji_pattern, word['text']):
            return 'EMOJ'

    def tag_emoticon(self, word, previous_words, next_words):
        if self.helper.is_punctuation(word):
            emoticon_pattern = re.compile('[:;=](?:-)?[)DPp\/]')
            if re.search(emoticon_pattern, word['text']):
                return 'EMOT'

    def tag_exclamation_mark(self, word, previous_words, next_words):
        if word['text'] == '!':
            return 'EXCL'

    def tag_hashtag(self, word, previous_words, next_words):
        if word['text'][0] == '#':
            return 'HASH'

    def tag_infinitives(self, word, previous_words, next_words):
        if self.helper.is_infinitive(word):
            return "INF"

    def tag_interjection(self, word, previous_words, next_words):
        if word['xpos'] == 'UH':
            return "UH"

    def tag_numeral(self, word, previous_words, next_words):
        if self.helper.is_numeral(word):
            return "NUM"

    def tag_laughter_acronyms(self, word, previous_word, next_words):
        if word['text'].lower() in self.patterns['laughter_acronyms']:
            return "LAUGH"

    def tag_possessive_pronoun(self, word, previous_words, next_words):
        if word['xpos'][:3] == 'PRP':
            return 'PRP'

    def tag_preposition(self, word, previous_words, next_words):
        if self.helper.is_preposition(word):
            return "PREP"

    def tag_proper_noun(self, word, previous_words, next_words):
        if word['xpos'][:3] == 'NNP':
            return "NNP"

    def tag_question_mark(self, word, previous_words, next_words):
        if word['text'] == '?':
            return 'QUES'

    def tag_quotation_mark(self, word, previous_words, next_words):
        if word['text'] == "'" or word['text'] == '"':
            return "QUOT"

    def tag_at(self, word, previous_words, next_words):
        if word['text'][0] == '@':
            return 'AT'

    def tag_subject_pronouns(self, word, previous_words, next_words):
        if self.helper.is_subject_pronoun(word):
            return "SBJP"

    def tag_url(self, word, previous_words, next_words):
        url_pattern = "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
        if re.search(url_pattern, word['text']):
            return "URL"

    def tag_wh_word(self, word, previous_words, next_words):
        if word['xpos'][0] == 'W':
            return "WH"

    def tag_indefinite_article(self, word, previous_words, next_words):
        if self.helper.is_indefinite_article(word):
            return "INDA"

    def tag_accusative_case(self, word, previous_words, next_words):
        if self.helper.is_accusative_case(word):
            return "ACCU"

    def tag_progressive_aspect(self, word, previous_words, next_words):
        if self.helper.is_progressive_aspect(word):
            return "PGAS"

    def tag_comparative(self, word, previous_words, next_words):
        if self.helper.is_comparative_adjective(word):
            return "CMADJ"

    def tag_superlative(self, word, previous_words, next_words):
        """ Noun (subject) + verb + the + superlative adjective + noun (object) """
        if self.helper.is_superlative_adjective(word):
            return "SPADJ"

    def tag_non_pos(self, word, previous_words, next_words):
        if self.helper.is_non_pos(word):
            return 'X'
