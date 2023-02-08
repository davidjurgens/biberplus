class WordTagger:
    def __init__(self, sentence, word, word_index, patterns_dict):
        """ Tag a sentence for the Biber tags that requires prior tags to complete and a specific order
            :param sentence: Tagged sentence from stanza
            :param word: Tagged word from stanza
            :param word_index: Position of the word in the sentence
            :param patterns_dict: Dictionary containing list of words for different patterns.
            e.g. public verbs, downtowners, etc.
            :return:
        """
        self.sentence = sentence
        self.word = word
        self.word_index = word_index,
        self.patterns_dict = patterns_dict
        self.fn_list_ordered = [
        ]

    def run_all(self):
        for fn in self.fn_list_ordered:
            fn()

    def tag_demp(self):
        """ The program tags as demonstrative pronouns the words those, this, these when they are
            followed by a verb (any tag starting with V) or auxiliary verb (modal verbs in the form of
            MD tags or forms of DO or forms of HAVE or forms of BE) or a punctuation mark or a WH
            pronoun or the word and. The word that is tagged as a demonstrative pronoun when it
            follows the said pattern or when it is followed by ‘s or is and, at the same time, it has not
            been already tagged as a TOBJ, TSUB, THAC or THVC.
        """
        pass
