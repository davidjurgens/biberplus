import re

from bibermda.tagger.tag_helper import TagHelper


class GrieveClarkeTagger:
    def __init__(self, tagged_words, token_normalization, patterns_dict):
        self.tagged_words = tagged_words
        self.token_normalization = token_normalization
        self.patterns = patterns_dict
        self.helper = TagHelper(patterns_dict)

    def run_all(self):
        # Batches of text by text normalization size
        # Get the full text and the list of
        # Check the list of upos
        return self.tagged_words

    def contains_fpp1(self):
        """ Any item of this list: I, me, us, my, we, our, myself, ourselves and their contractions.
        Tokenizer separates contractions """
        pass

    def contains_spp2(self):
        """ Any item of this list: I, me, us, my, we, our, myself, ourselves and their contractions.
        Tokenizer separates contractions """
        pass

    def contains_tpp3(self):
        pass

    def contains_accusative_case(self):
        pass

    def contains_articles(self):
        pass

    def contains_attributive_adjectives(self):
        pass

    def contains_auxillary_be(self):
        pass

    def contains_bema(self):
        pass

    def contains_bpya(self):
        pass

    def contains_capitalizations(self):
        pass

    def contains_comparatives(self):
        pass

    def contains_contrastive_conjunctions(self):
        pass

    def contains_coordinating_conjunctions(self):
        pass

    def contains_determiners(self):
        pass

    def contains_emoticons(self):
        pass

    def contains_exclamation_mark(self):
        pass

    def contains_existentials(self):
        pass

    def contains_hashtags(self):
        pass

    def contains_infinitives(self):
        pass

    def contains_initial_do(self):
        pass

    def contains_initial_verbs(self):
        pass

    def contains_interjections(self):
        pass

    def contains_nominalizations(self):
        pass

    def contains_numerals(self):
        pass

    def contains_object_pronouns(self):
        pass

    def contains_other_pronouns(self):
        pass

    def contains_passives(self):
        pass

    def contains_past_tense(self):
        pass

    def contains_perfect_aspect(self):
        pass

    def contains_place_adverbials(self):
        pass

    def contains_possessive_pronouns(self):
        pass

    def contains_predicative_adjectives(self):
        pass

    def contains_prepositions(self):
        pass

    def contains_proverb_do(self):
        pass

    def contains_progressive_aspect(self):
        pass

    def contains_proper_nouns(self):
        pass

    def contains_public_verbs(self):
        pass

    def contains_quantifiers(self, text):
        pattern = re.compile(
            r'\b(?:all|some|many|few|several|most|fewer|more|plenty of|lots of|a lot of|a few|a couple of)\b',
            re.IGNORECASE)
        return re.search(pattern, text, flags=re.IGNORECASE)

    def contains_question_do(self):
        pass

    def contains_question_mark(self, text):
        if re.search(r'\?', text):
            return "QUES"

    def contains_quotation_mark(self, text):
        if re.search(r'"', text):
            return "QUOT"

    def contains_subject_pronouns(self):
        pass

    def contains_synthetic_negation(self):
        pass

    def contains_urls(self, text):
        pattern = "https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"
        if re.search(pattern, text, flags=re.IGNORECASE):
            return "URL"

    def contains_wh_clause(self):
        pass

    def contains_wh_words(self):
        pass
