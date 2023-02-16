class WordTagger:
    def __init__(self, tagged_words, patterns_dict):
        """ Tag a sentence for the Biber tags that requires prior tags to complete and a specific order
            :param patterns_dict: Dictionary containing list of words for different patterns.
            e.g. public verbs, downtowners, etc.
            :return:
        """
        self.tagged_words = tagged_words
        self.patterns_dict = patterns_dict
        self.type_token_ratios = {}
        self.fn_list_ordered = [
        ]

    def run_all(self):
        for fn in self.fn_list_ordered:
            fn()

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

    def tag_wchl(self, word, word_index):
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

    def tag_presp(self, word, word_index):
        """ Present participial clauses. A punctuation mark is followed by a present participial form of a
        verb (VBG) followed by a preposition (PIN), a determiner (DT, QUAN, CD), a WH pronoun, a WH possessive
        pronoun (WP$), any WH word, any pronoun (PRP) or any adverb (RB).
        e.g. Stuffing his mouth with cookies, Joe ran out the door """
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

    def tag_wzpast(self, word, word_index):
        """ Past participial WHIZ deletion relatives. A noun (N) or quantifier pronoun (QUPR) followed by a
        past participial form of a verb (VBN) followed by a preposition (PIN) or an adverb (RB) or a form of BE.
        e.g. The solution produced by this process """
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
