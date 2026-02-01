import inspect
import re

import numpy as np

from biberplus.tagger.biber_run_order import RUN_ORDER
from biberplus.tagger.tag_helper import TagHelper


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
        """Run all tagger functions defined in this class."""

        def get_tagging_methods():
            """Get all methods that start with 'tag' and sort them by RUN_ORDER."""
            attrs = (getattr(self, name) for name in dir(self))
            methods = filter(inspect.ismethod, attrs)
            tag_methods = [m for m in methods if m.__name__.startswith("tag")]
            missing_methods = set([method.__name__ for method in tag_methods]) - set(
                RUN_ORDER
            )

            if missing_methods:
                raise ValueError(
                    f"Missing methods in RUN_ORDER: {', '.join(missing_methods)}"
                )

            return sorted(tag_methods, key=lambda x: RUN_ORDER.index(x.__name__))

        tag_methods = get_tagging_methods()

        for index, tagged_word in enumerate(self.tagged_words):
            # Get the context surrounding the word. Sets to None if unavailable
            previous_words = self.get_previous_n_words(index, n=4)
            next_words = self.get_next_n_words(index, n=4)
            self.current_index = index

            for tag_method in tag_methods:
                tag = tag_method(tagged_word, previous_words, next_words)
                if tag:
                    tagged_word["tags"].append(tag)

            self.update_doc_level_stats(tagged_word)

        self.mean_word_length = np.array(self.word_lengths).mean()

        return self.tagged_words

    """ Helper functions """

    def update_doc_level_stats(self, word):
        if self.helper.is_adverb(word):
            self.adverb_count += 1

        self.word_lengths.append(len(word["text"]))

    def get_previous_n_words(self, index, n):
        if index - n < 0:
            padding = [None] * (n - index)
            n -= len(padding)
            previous_n_words = self.tagged_words[index - n : n][::-1]
            previous_n_words.extend(padding)
            return previous_n_words
        else:
            return self.tagged_words[index - n : index][::-1]

    def get_next_n_words(self, index, n):
        next_n_words = self.tagged_words[index + 1 : index + n + 1]
        if len(next_n_words) < n:
            padding = [None] * (n - len(next_n_words))
            next_n_words.extend(padding)
        return next_n_words

    def get_phrase(self, next_n_words):
        curr_word_txt = self.tagged_words[self.current_index]["text"].lower()
        next_n_text = [w["text"].lower() for w in next_n_words]
        phrase = curr_word_txt + " " + " ".join(next_n_text)
        return phrase

    """ A) Tense and Aspect Markers """

    def tag_vbd(self, word, previous_words, next_words):
        """Past tense POS"""
        if word["xpos"] == "VBD":
            return "VBD"

    def tag_peas(self, word, previous_words, next_words):
        """Perfect aspect
        1) HAVE + (ADV) + (ADV) + VBD/VBN
        2) HAVE + N/PRO + VBN/VBD
        """
        if not self.helper.is_have(word):
            return

        # Direct verb after HAVE: HAVE + VBD/VBN
        if next_words[0] and next_words[0]["xpos"] in ["VBD", "VBN"]:
            return "PEAS"

        # One intervening word: HAVE + (ADV/N/PRO) + VBD/VBN
        if next_words[1] and next_words[1]["xpos"] in ["VBD", "VBN"]:
            if (
                self.helper.is_adverb(next_words[0])
                or self.helper.is_noun(next_words[0])
                or self.helper.is_pronoun(next_words[0])
            ):
                return "PEAS"

        # Two intervening adverbs: HAVE + (ADV) + (ADV) + VBD/VBN
        if next_words[2] and next_words[2]["xpos"] in ["VBD", "VBN"]:
            if self.helper.is_adverb(next_words[0]) and self.helper.is_adverb(
                next_words[1]
            ):
                return "PEAS"

    def tag_vprt(self, word, previous_words, next_words):
        """Present tense: VBP or VBZ tag"""
        if word["xpos"] in ["VBP", "VBZ"]:
            return "VPRT"

    """ B) PLACE and TIME Adverbials """

    def tag_place(self, word, previous_words, next_words):
        """Any item in the place adverbials list that is not a proper noun (NNP)"""
        if word["text"].lower() in self.patterns[
            "place_adverbials"
        ] and not self.helper.is_proper_noun(word):
            return "PLACE"

    def tag_time(self, word, previous_words, next_words):
        """Time adverbials with the exception: soon is not a time adverbial if it is followed by the word as"""
        word_text = word["text"].lower()

        if word_text == "soon" and next_words and next_words[0]["text"].lower() == "as":
            return

        if word_text in self.patterns["time_adverbials"]:
            return "TIME"

    """ C) Pronouns and pro-verbs """

    def tag_fpp1(self, word, previous_words, next_words):
        """Any item of this list: I, me, us, my, we, our, myself, ourselves and their contractions.
        Tokenizer separates contractionas"""
        if word["text"].lower() in self.patterns["first_person_pronouns"]:
            return "FPP1"

    def tag_spp2(self, word, previous_words, next_words):
        """Any item of this list: you, your, yourself, yourselves, thy, thee, thyself, thou"""
        if word["text"].lower() in self.patterns["second_person_pronouns"]:
            return "SPP2"

    def tag_tpp3(self, word, previous_words, next_words):
        """Any item of this list: she, he, they, her, him, them, his, their, himself, herself, themselves"""
        if word["text"].lower() in self.patterns["third_person_pronouns"]:
            return "TPP3"

    def tag_pit(self, word, previous_words, next_words):
        """Any pronoun it. Although not specified in Biber (1988), the present program also tags its and
        itself as "Pronoun it"."""
        if (
            self.helper.is_pronoun(word)
            and word["text"].lower() in self.patterns["pronoun_it"]
        ):
            return "PIT"

    def tag_inpr(self, word, previous_words, next_words):
        """Any item of this list: anybody, anyone, anything, everybody, everyone, everything, nobody,
        none, nothing, nowhere, somebody, someone, something"""
        if word["text"].lower() in self.patterns["indefinite_pronouns"]:
            return "INPR"

    def tag_demp(self, word, previous_words, next_words):
        """Tags as demonstrative pronouns the words: those, this, these, and that when they fit specific patterns."""

        if word["text"].lower() in self.patterns["demonstrative_pronouns"]:
            excluded_tags = {"TOBJ", "TSUB", "THAC", "THVC"}

            # If the word already has one of the excluded tags, do not process further
            if any(tag in word["tags"] for tag in excluded_tags):
                return

            if next_words[0]:
                following_word = next_words[0]["text"].lower()
                following_tag = next_words[0]["xpos"]
                is_followed_by_verb = self.helper.is_any_verb(next_words[0])
                is_followed_by_specifics = following_word in ["'s", "and"]
                is_followed_by_punct = self.helper.is_punctuation(next_words[0])

                # Changed to check any tag starting with 'W'
                is_followed_by_wh = following_tag.startswith("W")

                if any(
                    [
                        is_followed_by_verb,
                        is_followed_by_specifics,
                        is_followed_by_punct,
                        is_followed_by_wh,
                    ]
                ):
                    return "DEMP"

    def tag_prod(self, word, previous_words, next_words):
        """Pro-verb do. Any form of DO that is used as main verb and, therefore, excluding DO when used as
        auxiliary verb. The tagger tags as PROD any DO that is NOT in neither of the following patterns:
        (a) DO followed by a verb (any tag starting with V) or followed by adverbs (RB), negations and then
        a verb (V); (b) DO preceded by a punctuation mark or a WH pronoun"""
        if not self.helper.is_do(word):
            return
        # Exclude DO + Verb
        if next_words[0] and self.helper.is_verb(next_words[0]):
            return

        # Exclude DO + Adverb + Verb
        if (
            next_words[1]
            and self.helper.is_adverb(next_words[0])
            and self.helper.is_verb(next_words[1])
        ):
            return

        # Exclude PUNCT + DO and WHP + DO
        if previous_words[0] and (
            self.helper.is_punctuation(previous_words[0])
            or previous_words[0]["xpos"] == "WP"
        ):
            return

        # Exclude DO tokens immediately preceded by any WH word (e.g., "How")
        if previous_words[0] and previous_words[0]["xpos"].startswith("W"):
            return

        # Additional check for auxiliary DO at sentence start:
        # For example, in "Do you know the answer?" token "Do" is followed by a subject pronoun ("you")
        # and then a verb ("know"), so we should not tag it as PROD.
        if (
            previous_words[0] is None
            and next_words[0]
            and self.helper.is_subject_pronoun(next_words[0])
            and next_words[1]
            and self.helper.is_verb(next_words[1])
        ):
            return

        return "PROD"

    """ D) Questions """

    def tag_whqu(self, word, previous_words, next_words):
        """Direct WH-questions. Punctuation + WH word + auxiliary verb. Slightly modified to allow
        for intervening word between punctuation and WH word"""
        if word["xpos"][0] == "W" and next_words[0]:
            # Check if next token is auxiliary or (if not) the token after is auxiliary
            if self.helper.is_auxiliary(next_words[0]) or (
                len(next_words) > 1
                and next_words[1]
                and self.helper.is_auxiliary(next_words[1])
            ):
                if self.helper.is_punctuation(previous_words[0]) or (
                    previous_words[1] and self.helper.is_punctuation(previous_words[1])
                ):
                    return "WHQU"

    """ E) Nominal Forms """

    def tag_nomz(self, word, previous_words, next_words):
        suffixes = (
            "ity",
            "tion",
            "sion",
            "ment",
            "ness",
            "tions",
            "sions",
            "ments",
            "nesses",
        )
        if self.helper.is_noun(word) and word["text"].lower().endswith(suffixes):
            if word["text"].lower() not in self.patterns["nominalizations_stop_list"]:
                return "NOMZ"

    def tag_ger(self, word, previous_words, next_words):
        # If it's recognized as a verb in gerund form (VBG) *or* recognized as a noun,
        # and is 10+ letters ending in "ing" or "ings," then call it 'GER'
        if (
            (word["xpos"] == "VBG" or self.helper.is_noun(word))
            and len(word["text"]) >= 10
            and word["text"].lower().endswith(("ing", "ings"))
        ):
            return "GER"

    def tag_nn(self, word, previous_words, next_words):
        """Total other nouns. Any noun not tagged as a nominalisation or a gerund. Plural nouns (NNS) and
        proper nouns (NNP and NNPS) tags are changed to NN and included in this count"""
        if (
            self.helper.is_noun(word)
            and "NOMZ" not in word["tags"]
            and "GER" not in word["tags"]
        ):
            return "NN"

    """ F) Passives """

    def tag_pass(self, word, previous_words, next_words):
        """Agentless passives are tagged for 2 patterns. First, any form BE + 1-2 optional RBs + (VBD|VBN).
        Second any form BE + nominal form (noun|pronoun) + (VBN). Following original Biber which does not allow
        for intervening negation in this pattern"""

        if not self.helper.is_be(word):
            return

        def is_past_verb(word):
            return word and word["xpos"] in ["VBN", "VBD"]

        # BE + VBN/VBD
        if is_past_verb(next_words[0]):
            return "PASS"

        # BE + be + VBN/VBD (handle progressive/passive constructions, e.g. "is being renovated")
        if (
            next_words[0]
            and self.helper.is_be(next_words[0])
            and is_past_verb(next_words[1])
        ):
            return "PASS"

        # BE + ADV + VBN/VBD
        if self.helper.is_adverb(next_words[0]) and is_past_verb(next_words[1]):
            return "PASS"

        # BE + N/PRO + VBN/VBD
        if (
            self.helper.is_noun(next_words[0]) or self.helper.is_pronoun(next_words[0])
        ) and is_past_verb(next_words[1]):
            return "PASS"

        # BE + ADV + ADV + VBN/VBD
        if (
            self.helper.is_adverb(next_words[0])
            and self.helper.is_adverb(next_words[1])
            and is_past_verb(next_words[2])
        ):
            return "PASS"

    def tag_bypa(self, word, previous_words, next_words):
        """By-passives. PASS are found and the preposition by follows it"""
        if "PASS" in word["tags"]:
            for next_word in next_words[:4]:  # Check up to the next 4 words
                if next_word and next_word["text"].lower() == "by":
                    return "BYPA"

    """ G) Stative Forms"""

    def tag_bema(self, word, previous_words, next_words):
        """Be as main verb (BEMA): BE followed by a (DT), (PRP$) or a (PIN) or an adjective (JJ).
        Allows adverbs or negations to appear between the verb BE and the rest of the pattern.
        """

        if not self.helper.is_be(word):
            return None

        # If immediately preceded by an existential "there", do not tag BEMA.
        if previous_words[0] and previous_words[0]["xpos"] == "EX":
            return None

        # If the next word is a past-participle (indicative of a passive construction), do not tag BEMA.
        if next_words[0] and next_words[0]["xpos"] in ["VBN", "VBD"]:
            return None

        valid_tags = ["DT", "PRP$", "JJ", "JJR"]

        # Directly follows BE
        if next_words[0] and (
            next_words[0]["xpos"] in valid_tags
            or self.helper.is_preposition(next_words[0])
        ):
            return "BEMA"

        # Preceded by an adverb or negation - check next_words[0] then next_words[1]
        if (
            next_words[1]
            and next_words[0]
            and (
                self.helper.is_adverb(next_words[0])
                or self.tag_xx0(next_words[0], None, None)
            )
        ):
            if next_words[1]["xpos"] in valid_tags or self.helper.is_preposition(
                next_words[1]
            ):
                return "BEMA"

    def tag_ex(self, word, previous_words, next_words):
        """Existential there from the POS tags"""
        if word["xpos"] == "EX":
            return "EX"

    """ H) Subordination Features """

    def tag_thvc(self, word, previous_words, next_words):
        if word["text"].lower() != "that":
            return

        def is_rel_verb(curr_word):
            txt = curr_word["text"].lower()
            return txt in (
                self.patterns["public_verbs"]
                | self.patterns["private_verbs"]
                | self.patterns["suasive_verbs"]
            )

        # Condition 1: Preceded by certain conjunctions or punctuation, followed by specific tags
        if previous_words[0] and next_words[0]:
            prev_word = previous_words[0]["text"].lower()
            following_tags = ["DT", "CD", "PRP", "NNS", "NNP"]
            if prev_word in [
                "and",
                "nor",
                "but",
                "or",
                "also",
            ] or self.helper.is_punctuation(previous_words[0]):
                if next_words[0]["xpos"] in following_tags or self.helper.is_quantifier(
                    next_words[0]
                ):
                    return "THVC"

            # Condition 2: Preceded by specific verbs, followed by anything except verbs, punctuation or 'and'
            if is_rel_verb(previous_words[0]) or "SMP" in previous_words[0]["tags"]:
                exclusions = [
                    self.helper.is_any_verb,
                    self.helper.is_punctuation,
                    lambda w: w["text"].lower() == "and",
                ]
                if not any(func(next_words[0]) for func in exclusions):
                    return "THVC"

            # Condition 3: Preceded by specific verbs and a preposition, and up to four non-noun words
            if self.helper.is_noun(previous_words[0]):
                if (
                    previous_words[2]
                    and is_rel_verb(previous_words[2])
                    and self.helper.is_preposition(previous_words[1])
                ):
                    return "THVC"

                # Check for up to 4 intervening words that are not nouns
                for i in range(1, 5):
                    prev_n_words = self.get_previous_n_words(
                        self.current_index, n=i + 3
                    )
                    if (
                        prev_n_words[i + 2]
                        and is_rel_verb(prev_n_words[0])
                        and self.helper.is_preposition(prev_n_words[1])
                    ):
                        if not any(
                            self.helper.is_noun(w) for w in prev_n_words[2 : i + 1]
                        ):
                            return "THVC"

    def tag_thac(self, word, previous_words, next_words):
        """That adjective complements. That preceded by an adjective (JJ or a predicative adjective, PRED)."""
        if word["text"].lower() == "that" and previous_words[0]:
            if (
                self.helper.is_adjective(previous_words[0])
                or "PRED" in previous_words[0]["tags"]
            ):
                return "THAC"

    def tag_whcl(self, word, previous_words, next_words):
        """WH-clauses. any public, private or suasive verb followed by any WH word, followed by a word that is
        NOT an auxiliary (tag MD for modal verbs, or a form of DO, or a form of HAVE, or a form of BE)."""
        verb_tags = {"PUBV", "PRIV", "SUAV"}

        if (
            word["xpos"].startswith("W")
            and previous_words[0]
            and any(tag in previous_words[0]["tags"] for tag in verb_tags)
        ):
            if next_words[0] and not self.helper.is_auxiliary(next_words[0]):
                return "WHCL"

    def tag_to(self, word, previous_words, next_words):
        """Infinitives: POS tag TO that are not a preposition"""
        if word["xpos"] == "TO":
            # Prepositions are 'to's followed by
            filter_preps = [
                "IN",
                "CD",
                "DT",
                "JJ",
                "PRP$",
                "WP$",
                "WDT",
                "WP",
                "WRB",
                "PDT",
                "N",
                "NNS",
                "NP",
                "NPs",
                "PRP",
            ]
            if next_words[0] and next_words[0]["xpos"] not in filter_preps:
                return "TO"

    def tag_presp(self, word, previous_words, next_words):
        """Present participial clause.
        - A word with 'VBG' as xpos.
        - Preceded by a punctuation mark.
        - Followed by specific xpos values (e.g., PIN, DT, QUAN, etc.)."""

        valid_next_tags = {
            "PIN",
            "DT",
            "QUAN",
            "CD",
            "WH",
            "WP",
            "WP$",
            "PRP",
            "RB",
            "WRB",
        }

        if (
            word["xpos"] == "VBG"
            and previous_words[0]
            and self.helper.is_punctuation(previous_words[0])
        ):
            if next_words[0] and next_words[0]["xpos"] in valid_next_tags:
                return "PRESP"

    def tag_pastp(self, word, previous_words, next_words):
        """Past partcipial clauses: punctuation followed by VBN -> PIN or RB
        e.g. 'Built' in a single week, the house would stand for fifty years"""
        if (
            word["xpos"] == "VBN"
            and previous_words[0]
            and self.helper.is_punctuation(previous_words[0])
            and next_words[0]
            and (
                self.helper.is_adverb(next_words[0])
                or self.tag_pin(next_words[0], None, None)
            )
        ):
            return "PASTP"

    def tag_wzpast(self, word, previous_words, next_words):
        """Past participial WHIZ deletion relatives. Characterized by:
        - A word with 'VBN' as xpos.
        - Preceded by a noun or a quantifier pronoun.
        - Followed by a preposition, an adverb, or a form of the verb "BE"."""
        if (
            word["xpos"] == "VBN"
            and previous_words[0]
            and (
                self.helper.is_noun(previous_words[0])
                or self.helper.is_quantifier_pronoun(previous_words[0])
            )
            and next_words[0]
            and (
                self.helper.is_preposition(next_words[0])
                or self.helper.is_adverb(next_words[0])
                or self.helper.is_be(next_words[0])
            )
        ):
            return "WZPAST"

    def tag_wzpres(self, word, previous_words, next_words):
        """Present participial WHIZ deletion relatives: VBG preceded by an NN
        e.g. the 'causing' this decline' is"""
        if word["xpos"] == "VBG":
            if previous_words[0] and previous_words[0]["xpos"].startswith("NN"):
                return "WZPRES"

    def tag_tsub(self, word, previous_words, next_words):
        """That relative clauses on subject position. Criteria:
        - The word is 'that'.
        - Preceded by a noun.
        - Followed by a verb or an auxiliary verb, possibly with an intervening adverb or negation."""
        if word["text"].lower() != "that":
            return None

        if (
            previous_words[0]
            and next_words[0]
            and self.helper.is_noun(previous_words[0])
        ):
            # Check if following word is a verb or auxiliary
            if self.helper.is_verb(next_words[0]) or self.helper.is_auxiliary(
                next_words[0]
            ):
                return "TSUB"

            # Allow for intervening RB or XXO.
            # Make sure next_words[0] exists before passing it to tag_xx0.
            if (
                next_words[1]
                and next_words[0]
                and (
                    self.tag_xx0(next_words[0], None, None)
                    or self.helper.is_adverb(next_words[0])
                )
            ):
                if self.helper.is_any_verb(next_words[1]):
                    return "TSUB"

    def tag_tobj(self, word, previous_words, next_words):
        """That relative clauses on object position. Criteria:
        - The word is 'that'.
        - Preceded by a noun.
        - Followed by one of: determiner, subject form of a personal pronoun,
          possessive pronoun, the pronoun 'it', an adjective, plural noun,
          proper noun, or possessive noun."""
        if (
            word["text"].lower() != "that"
            or not previous_words[0]
            or not self.helper.is_noun(previous_words[0])
        ):
            return

        # Tags for the word following 'that'
        allowed_following_tags = {"DT", "CD", "PRP", "PRP$", "NNS", "NNP", "JJ"}
        next_word = next_words[0]

        if next_word and (
            next_word["xpos"] in allowed_following_tags
            or self.helper.is_quantifier(next_word)
            or self.tag_pit(next_word, previous_words[1:], next_words[1:])
            or self.helper.is_possesive_pronoun(next_word)
        ):
            return "TOBJ"

    def tag_whsub(self, word, previous_words, next_words):
        """WH relative clauses on subject position. Criteria:
        - Starts with a WH pronoun.
        - Preceded by a word that's NOT a form of ASK or TELL and a noun.
        - Followed by a verb or an auxiliary verb, possibly with an intervening adverb or negation"""
        if (
            word["xpos"][0] == "W"
            and previous_words[1]
            and previous_words[1]["text"].lower() not in self.patterns["ask_tell"]
            and self.helper.is_noun(previous_words[0])
        ):
            # NOT ASK/TELL -> Noun -> WP -> (RB/XXO) -> Verb
            if next_words[0] and (
                self.helper.is_any_verb(next_words[0])
                or (
                    next_words[1]
                    and (
                        self.helper.is_adverb(next_words[0])
                        or self.tag_xx0(next_words[0], None, None)
                    )
                    and self.helper.is_any_verb(next_words[1])
                )
            ):
                return "WHSUB"

    def tag_whobj(self, word, previous_words, next_words):
        """WH relative clauses on object position. Criteria:
        - Starts with a WH pronoun.
        - Not preceded by a form of ASK or TELL and followed by a noun.
        - Not followed by an adverb, a negation, a verb or an auxiliary verb."""
        if not word["xpos"].startswith("W"):
            return

        if next_words[0] and not (
            self.helper.is_adverb(next_words[0])
            or self.helper.is_any_verb(next_words[0])
            or self.tag_xx0(next_words[0], previous_words[1:], next_words[1:])
        ):
            if (
                previous_words[2]
                and previous_words[2]["text"].lower() not in self.patterns["ask_tell"]
                and self.helper.is_noun(previous_words[0])
            ):
                return "WHOBJ"

    def tag_pire(self, word, previous_words, next_words):
        """Pied-piping relatives clauses. Any preposition (PIN) followed by whom, who, whose or which"""
        if word["text"].lower() in ["whom", "who", "whose", "which"]:
            if previous_words[0] and "PIN" in previous_words[0]["tags"]:
                return "PIRE"

    def tag_sere(self, word, previous_words, next_words):
        """Sentence relatives. Everytime a punctuation mark is followed by the word which"""
        if word["text"].lower() == "which":
            if previous_words[0] and self.helper.is_punctuation(previous_words[0]):
                return "SERE"

    def tag_caus(self, word, previous_words, next_words):
        """Any occurrence of the word because"""
        if word["text"].lower() == "because":
            return "CAUS"

    def tag_conc(self, word, previous_words, next_words):
        """Any occurrence of the words although, though, tho"""
        if word["text"].lower() in self.patterns["concessive_adverbial_subordinators"]:
            return "CONC"

    def tag_cond(self, word, previous_words, next_words):
        """Any occurrence of the words if or unless"""
        if word["text"].lower() in self.patterns["conditional_adverbial_subordinators"]:
            return "COND"

    def tag_osub(self, word, previous_words, next_words):
        """Other adverbial subordinators. Any occurrence of the OSUB words. For multi-word units only tag the first"""
        if word["text"].lower() in self.patterns["other_adverbial_subordinators"]:
            return "OSUB"

        if not next_words:
            return

        # 2 word case
        if next_words[0]:
            phrase = self.get_phrase(next_words[:1])
            if phrase in self.patterns["other_adverbial_subordinators"]:
                return "OSUB"

        # 3 word case
        if next_words[1]:
            phrase = self.get_phrase(next_words[:2])
            if phrase in self.patterns["other_adverbial_subordinators"]:
                return "OSUB"

            # So that and such that cases
            if phrase in ["so that", "such that"]:
                # Cannot be followed by a noun or adjective
                if not (
                    self.helper.is_noun(next_words[1])
                    or self.helper.is_adjective(next_words[1])
                ):
                    return "OSUB"

    """ I) Prepositional Phrases, Adjectives, and Adverbs"""

    def tag_pin(self, word, previous_words, next_words):
        """Total prepositional phrases"""
        if self.helper.is_preposition(word):
            return "PIN"

    def tag_jj(self, word, previous_words, next_words):
        """Attributive adjectives"""
        if word["xpos"] in ["JJ", "JJR", "JJS"]:
            return "JJ"

    def tag_pred(self, word, previous_words, next_words):
        """Predicative adjectives. Any form of BE followed by an adjective (JJ) followed by a word that is NOT
        another adjective, an adverb (RB) or a noun (N). If any adverb or negation is intervening between the
        adjective and the word after it, the tag is still assigned.
        An adjective is tagged as predicative if it is
        preceded by another predicative adjective followed by a phrasal coordinator e.g. the horse is big and fast"""

        if not self.helper.is_adjective(word):
            return None

        # Handle direct BE->ADJ case
        if previous_words[0] and self.helper.is_be(previous_words[0]):
            if next_words[0] and not (
                self.helper.is_adjective(next_words[0])
                or self.helper.is_adverb(next_words[0])
                or self.helper.is_noun(next_words[0])
            ):
                return "PRED"

            # Allow for intervening negation/adverb. Ensure next_words[0] exists.
            if (
                len(next_words) > 1
                and next_words[0]
                and (
                    self.tag_xx0(next_words[0], None, None)
                    or self.helper.is_adverb(next_words[0])
                )
            ):
                if next_words[1] and not (
                    self.helper.is_adjective(next_words[1])
                    or self.helper.is_noun(next_words[1])
                    or self.helper.is_adverb(next_words[1])
                ):
                    return "PRED"

        # Handle phrasal coordinator case
        if (
            previous_words[1]
            and "PRED" in previous_words[1]["tags"]
            and "PHC" in previous_words[0]["tags"]
        ):
            return "PRED"

    def tag_rb(self, word, previous_words, next_words):
        """Any adverb i.e. POS tags RB, RBS, RBR, WRB"""
        if self.helper.is_adverb(word):
            return "RB"

    """ J) Lexical Specificity """

    def compute_type_token_ratio(self):
        uniq_vocab = set()

        for i in range(self.ttr_n):
            uniq_vocab.add(self.tagged_words[i]["text"].lower())

        self.ttr = len(uniq_vocab) / self.ttr_n

    """ K) Lexical Classes """

    def tag_conj(self, word, previous_words, next_words):
        """Conjuncts find any item in the conjuncts list with preceding punctuation.
        Only the first word is tagged.
        """
        word_text = word["text"].lower()
        if word_text in self.patterns["conjucts"]:
            return "CONJ"

        # Specific words that should be tagged as CONJ when preceded by punctuation
        punctuation_sensitive_conjuncts = ["altogether", "rather"]
        if (
            previous_words
            and previous_words[0]
            and self.helper.is_punctuation(previous_words[0])
            and word_text in punctuation_sensitive_conjuncts
        ):
            return "CONJ"

    def tag_dwnt(self, word, previous_words, next_words):
        """Any instance of the words in the downtowners list"""
        if word["text"].lower() in self.patterns["downtoners"]:
            return "DWNT"

    def tag_amp(self, word, previous_words, next_words):
        """Any instance of the items in the amplifiers list"""
        if word["text"].lower() in self.patterns["amplifiers"]:
            return "AMP"

    def tag_dpar(self, word, previous_words, next_words):
        """Discourse particle: the words well, now, anyhow, anyways preceded by a punctuation mark"""
        if word["text"].lower() in self.patterns["discourse_particles"]:
            if previous_words[0] and self.helper.is_punctuation(previous_words[0]):
                return "DPAR"

    def tag_hdg(self, word, previous_words, next_words):
        """Hedges. Any hedge token. In cases of multi-word units such as more or less, only the first word is
        tagged as HDG. For the terms sort of and kind of these two items must be preceded by a determiner (DT),
        a quantifier (QUAN), a cardinal number (CD), an adjective (JJ or PRED), a possessive pronouns (PRP$) or
        WH word (see entry on WH-questions)
        """

        # One word hedges
        if word["text"].lower() in self.patterns["hedges"]:
            return "HDG"

        # Two word hedges
        if next_words[0]:
            phrase = self.get_phrase(next_words[:1])
            # Handle kind of / sort of case
            if phrase in ["kind of", "sort of"]:
                if previous_words[0] and (
                    previous_words[0]["xpos"] in ["DT", "CD", "PRP$"]
                    or previous_words[0]["xpos"][0] == "W"
                    or self.helper.is_quantifier(word)
                    or self.helper.is_adjective(previous_words[0])
                ):
                    return "HDG"
            elif phrase in self.patterns["hedges"]:
                return "HDG"

        # Three word hedges
        if next_words[1]:
            phrase = self.get_phrase(next_words[:2])
            if phrase in self.patterns["hedges"]:
                return "HDG"

    def tag_emph(self, word, previous_words, next_words):
        """Emphatics. Tags words and phrases that convey emphasis. Criteria:
        - Any word in the emphatics list.
        - Real+adjective, so+adjective.
        - Any form of DO followed by a verb.
        - Multi-word units such as "a lot", "for sure", "such a" where only the first word is tagged.
        """

        # Single word emphatics
        if word["text"].lower() in self.patterns["emphatics"]:
            return "EMPH"

        # Check for the availability of next word.
        if not next_words[0]:
            return None

        # Two-word emphatics: real+adjective and so+adjective
        if word["text"].lower() in ["real", "so"] and self.helper.is_adjective(
            next_words[0]
        ):
            return "EMPH"

        # DO form followed by a verb
        if self.helper.is_do(word) and self.helper.is_verb(next_words[0]):
            return "EMPH"

        # Other two-word emphatics in patterns
        phrase = self.get_phrase(next_words[:1])
        if phrase in self.patterns["emphatics"]:
            return "EMPH"

    def tag_demo(self, word, previous_words, next_words):
        """Demonstratives. words that, this, these, those have not been
        tagged as either DEMP, TOBJ, TSUB, THAC, or THVC"""

        is_demonstrative = word["text"].lower() in self.patterns["demonstratives"]
        has_invalid_tags = any(
            tag in word["tags"] for tag in ["DEMP", "TOBJ", "TSUB", "THAC", "THVC"]
        )

        if is_demonstrative and not has_invalid_tags:
            return "DEMO"

    """ L) Modals """

    def tag_pomd(self, word, previous_words, next_words):
        """The possibility modals listed by Biber (1988): can, may, might, could"""
        if word["text"].lower() in self.patterns["possibility_modals"]:
            return "POMD"

    def tag_nemd(self, word, previous_words, next_words):
        """The necessity modals listed by Biber (1988): ought, should, must."""
        if word["text"].lower() in self.patterns["necessity_modals"]:
            return "NEMD"

    def tag_prmd(self, word, previous_words, next_words):
        """Predictive modals. will, would, shall and their contractions: 'd_MD, ll_MD, wo_MD, sha_MD"""
        if (
            word["text"].lower() in self.patterns["predictive_modals"]
            and word["xpos"] == "MD"
        ):
            return "PRMD"

    """ M) Specialized Verb Classes """

    def tag_pubv(self, word, previous_words, next_words):
        """Any item in the public verbs list"""
        if word["text"].lower() in self.patterns["public_verbs"]:
            return "PUBV"

    def tag_priv(self, word, previous_words, next_words):
        """Any item in the private verbs list"""
        if word["text"].lower() in self.patterns["private_verbs"]:
            return "PRIV"

    def tag_suav(self, word, previous_words, next_words):
        """Any item in the suasive verbs list"""
        if word["text"].lower() in self.patterns["suasive_verbs"]:
            return "SUAV"

    def tag_smp(self, word, previous_words, next_words):
        """Any occurrence of the forms of the two verbs seem and appear"""
        if word["text"].lower() in self.patterns["seem_appear"]:
            return "SMP"

    """ N) Reduced forms and dispreferred structures """

    def tag_cont(self, word, previous_words, next_words):
        text = word["text"].lower()
        if "n't" in text:
            return "CONT"
        elif text.startswith("'") and text not in {"'s"}:
            return "CONT"

    def tag_thatd(self, word, previous_words, next_words):
        """Subordinator that deletion"""

        txt = word["text"].lower()
        if txt not in (
            self.patterns["public_verbs"]
            | self.patterns["private_verbs"]
            | self.patterns["suasive_verbs"]
        ):
            return
        if next_words[0] and (
            self.tag_demp(next_words[0], previous_words[1:], next_words[1:])
            or next_words[0]["text"].lower() in self.patterns["subject_pronouns"]
        ):
            return "THATD"

        # PUBV|PRIV|SUAV + PRO|N + V|AUX
        if next_words[1]:
            if self.helper.is_pronoun(next_words[0]) or self.helper.is_noun(
                next_words[0]
            ):
                if self.helper.is_any_verb(next_words[1]):
                    return "THATD"

        # PUBV|PRIV|SUAV + JJ|PRED|ADV|DT|QUAN|CD|PRP$ + N + V|AUXV
        if next_words[2]:
            next_word_2 = next_words[2]
            if (
                next_word_2["xpos"] in ["JJ", "DT", "CD", "PRP$"]
                or self.helper.is_adverb(next_word_2)
                or self.tag_pred(next_words[0], [word] + previous_words, next_words[1:])
                or self.helper.is_quantifier(next_words[0])
            ):
                if self.helper.is_noun(next_words[1]) and (
                    self.helper.is_verb(next_word_2)
                    or self.helper.is_auxiliary(next_word_2)
                ):
                    return "THATD"

        # PUBV|PRIV|SUAV + JJ|PRED|ADV|DT|QUAN|CD|PRP$ + (ADJ) + N + V|AUXV
        if next_words[3]:
            if (
                next_words[0]["xpos"] in ["JJ", "DT", "CD", "PRP$"]
                or self.helper.is_adverb(next_words[0])
                or self.tag_pred(next_words[0], previous_words[1:], next_words[1:])
                or self.helper.is_quantifier(next_words[0])
            ):
                if (
                    self.helper.is_adjective(next_words[1])
                    and self.helper.is_noun(next_words[2])
                    and self.helper.is_any_verb(next_words[3])
                ):
                    return "THATD"

    def tag_stpr(self, word, previous_words, next_words):
        """Stranded preposition. Preposition followed by a punctuation mark.
        Update from Biber: can't be the word besides. E.g. the candidates I was thinking 'of',"""
        if self.helper.is_preposition(word) and word["text"].lower() != "besides":
            if next_words[0] and self.helper.is_punctuation(next_words[0]):
                return "STPR"

    def tag_spin(self, word, previous_words, next_words):
        """Split infinitives. Every time an infinitive marker to is followed by one or two adverbs and
        a verb base form. e.g. he wants to convincingly prove that"""
        if word["text"].lower() != "to":
            return

        # Check if the next word is an adverb
        if not next_words[0] or not self.helper.is_adverb(next_words[0]):
            return

        # TO + 1 adverb + VB
        if next_words[1] and next_words[1]["xpos"] == "VB":
            return "SPIN"

        # TO + 2 adverbs + VB
        if (
            next_words[2]
            and self.helper.is_adverb(next_words[1])
            and next_words[2]["xpos"] == "VB"
        ):
            return "SPIN"

    def tag_spau(self, word, previous_words, next_words):
        """
        Split auxiliaries. Auxiliary (any modal verb MD, or any form of DO, or any form of BE, or any
        form of HAVE) is followed by one or two adverbs and a verb base form.
        Note: The existing tagger also uses VBN in addition to verb base form.
        """
        if not self.helper.is_auxiliary(word):
            return

        # AUX + 1 adverb + VB
        if (
            next_words[1]
            and self.helper.is_adverb(next_words[0])
            and next_words[1]["xpos"][:2] == "VB"
        ):
            return "SPAU"

        # AUX + 2 adverbs + VB
        if (
            next_words[2]
            and self.helper.is_adverb(next_words[0])
            and self.helper.is_adverb(next_words[1])
            and next_words[2]["xpos"][:2] == "VB"
        ):
            return "SPAU"

    """ O) Coordination """

    def tag_phc(self, word, previous_words, next_words):
        """Phrasal coordination. Any 'and' followed by the same tag if the tag is in (adverb, adjective, verb, noun)"""
        if word["text"].lower() == "and" and previous_words[0] and next_words[0]:
            if previous_words[0]["upos"] == next_words[0]["upos"]:
                if (
                    self.helper.is_adverb(previous_words[0])
                    or self.helper.is_adjective(previous_words[0])
                    or self.helper.is_verb(previous_words[0])
                    or self.helper.is_noun(previous_words[0])
                    or self.helper.is_proper_noun(previous_words[0])
                ):
                    return "PHC"

    def tag_andc(self, word, previous_words, next_words):
        """Independent clause coordination. Assigned to the word and when it is found in one of the following
        patterns:
        (1) preceded by a comma and followed by it, so, then, you, there + BE, or a demonstrative pronoun
        (DEMP) or the subject forms of a personal pronouns;
        (2) preceded by any punctuation;
        (3) followed by a WH pronoun or any WH word, an adverbial subordinator (CAUS, CONC, COND, OSUB) or a
        discourse particle (DPAR) or a conjunct (CONJ)"""
        if word["text"].lower() != "and":
            return

        # Condition (3): Check if "and" is followed by trigger words or punctuation signals
        if next_words[0]:
            triggers = set(["it", "so", "then", "you"])
            if self.patterns.get("demonstrative_pronouns"):
                triggers.update(
                    [w.lower() for w in self.patterns["demonstrative_pronouns"]]
                )
            if self.patterns.get("subject_pronouns"):
                triggers.update([w.lower() for w in self.patterns["subject_pronouns"]])
            if next_words[0]["text"].lower() in triggers:
                return "ANDC"
            if (
                next_words[0]["text"].lower() == "there"
                and next_words[1]
                and self.helper.is_be(next_words[1])
            ):
                return "ANDC"
            if next_words[0]["xpos"].startswith("W"):
                return "ANDC"
            if (
                self.tag_caus(next_words[0], previous_words[1:], next_words[1:])
                or self.tag_conc(next_words[0], previous_words[1:], next_words[1:])
                or self.tag_cond(next_words[0], previous_words[1:], next_words[1:])
                or self.tag_osub(next_words[0], previous_words[1:], next_words[1:])
                or self.tag_dpar(next_words[0], previous_words[1:], next_words[1:])
                or self.tag_conj(next_words[0], previous_words[1:], next_words[1:])
            ):
                return "ANDC"

        # Condition (2): If the token immediately before "and" is punctuation (or its text ends with a punctuation mark)
        if previous_words[0]:
            prev_text = previous_words[0]["text"]
            if self.helper.is_punctuation(previous_words[0]) or (
                prev_text and prev_text[-1] in [",", ";", ":"]
            ):
                return "ANDC"
        return

    """ P) Negation """

    def tag_xx0(self, word, previous_words, next_words):
        """Analytic negation: word 'not' and to the item n't_RB"""
        if not word:  # or `if word is None:`
            return None
        if word["text"].lower() in self.patterns["analytic_negation"]:
            return "XX0"

    def tag_syne(self, word, previous_words, next_words):
        """Synthetic negation: (no, neither, nor) followed by an adjective, noun, or proper noun"""
        if (
            not next_words[0]
            or word["text"].lower() not in self.patterns["synthetic_negations"]
        ):
            return
        next_word_conditions = (
            self.helper.is_adjective(next_words[0]),
            self.helper.is_noun(next_words[0]),
            self.helper.is_proper_noun(next_words[0]),
        )

        if any(next_word_conditions):
            return "SYNE"

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
        if word["text"].lower() in self.patterns["be"] and self.helper.is_auxiliary(
            word
        ):
            return "AUXB"

    def tag_capitalizations(self, word, previous_words, next_words):
        if word["text"][0].isupper():
            return "CAP"

    def tag_subordinating_conjunctions(self, word, previous_words, next_words):
        if self.helper.is_subordinating_conjunction(word):
            return "SCONJ"

    def tag_coordinating_conjunctions(self, word, previous_words, next_words):
        if self.helper.is_coordinating_conjunction(word):
            return "CCONJ"

    def tag_determiners(self, word, previous_words, next_words):
        if self.helper.is_determiner(word):
            return "DET"

    def tag_emoji(self, word, previous_words, next_words):
        emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
        if re.search(emoji_pattern, word["text"]):
            return "EMOJ"

    def tag_emoticon(self, word, previous_words, next_words):
        if self.helper.is_punctuation(word):
            emoticon_pattern = re.compile("[:;=](?:-)?[)DPp\/]")
            if re.search(emoticon_pattern, word["text"]):
                return "EMOT"

    def tag_exclamation_mark(self, word, previous_words, next_words):
        if word["text"] == "!":
            return "EXCL"

    def tag_hashtag(self, word, previous_words, next_words):
        if word["text"][0] == "#":
            return "HASH"

    def tag_infinitives(self, word, previous_words, next_words):
        if self.helper.is_infinitive(word):
            return "INF"

    def tag_interjection(self, word, previous_words, next_words):
        if word["xpos"] == "UH":
            return "UH"

    def tag_numeral(self, word, previous_words, next_words):
        if self.helper.is_numeral(word):
            return "NUM"

    def tag_laughter_acronyms(self, word, previous_word, next_words):
        if word["text"].lower() in self.patterns["laughter_acronyms"]:
            return "LAUGH"

    def tag_possessive_pronoun(self, word, previous_words, next_words):
        standalone_possessives = {"mine", "yours", "his", "hers", "ours", "theirs"}
        if word["xpos"][:3] == "PRP" or word["text"].lower() in standalone_possessives:
            return "PRP"

    def tag_preposition(self, word, previous_words, next_words):
        if self.helper.is_preposition(word):
            return "PREP"

    def tag_proper_noun(self, word, previous_words, next_words):
        if word["xpos"][:3] == "NNP":
            return "NNP"

    def tag_question_mark(self, word, previous_words, next_words):
        if word["text"] == "?":
            return "QUES"

    def tag_quotation_mark(self, word, previous_words, next_words):
        if word["text"] == "'" or word["text"] == '"':
            return "QUOT"

    def tag_at(self, word, previous_words, next_words):
        if word["text"][0] == "@":
            return "AT"

    def tag_subject_pronouns(self, word, previous_words, next_words):
        if self.helper.is_subject_pronoun(word):
            return "SBJP"

    def tag_url(self, word, previous_words, next_words):
        url_pattern = "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
        if re.search(url_pattern, word["text"]):
            return "URL"

    def tag_wh_word(self, word, previous_words, next_words):
        if word["xpos"][0] == "W":
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
        """Noun (subject) + verb + the + superlative adjective + noun (object)"""
        if self.helper.is_superlative_adjective(word):
            return "SPADJ"

    def tag_non_pos(self, word, previous_words, next_words):
        if self.helper.is_non_pos(word):
            return "X"
