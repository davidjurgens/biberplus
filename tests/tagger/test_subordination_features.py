import unittest

import spacy

from biberplus.tagger import tag_text


class TestSubordinationFeatureFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = spacy.load(
            "en_core_web_sm", disable=["parser", "lemmatizer", "ner"]
        )

    def test_thvc(self):
        text = "I've read a few of these reviews and think that Fisher Price must have a quality control issue ."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # That should be tagged as a THVC
        self.assertIn("THVC", tagged_words[10]["tags"])

    def test_thvc_case_followed_by_determiner(self):
        text = "I've read a few of these reviews and think that Fisher Price must have a quality control issue ."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # 'That' should be tagged as a THVC
        self.assertIn("THVC", tagged_words[10]["tags"])

    def test_thvc_case_preceded_by_noun_and_preposition(self):
        text = "I heard that they had a great time at the party ."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # 'That' should be tagged as a THVC
        self.assertIn("THVC", tagged_words[2]["tags"])

    def test_thac(self):
        text = "twice a day for 20 minutes per use . Disappointing that it failed so quickly . I have now owned"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # That should be tagged as a THAC
        self.assertIn("THAC", tagged_words[10]["tags"])

    def test_whcl(self):
        text = "it gingerly with his foot . How could anyone know what to do with an assortment like that ? Perhaps"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # What should be tagged as WHCL
        self.assertIn("WHCL", tagged_words[10]["tags"])

    def test_presp(self):
        text = "practice and for that it is a good resource . Knowing why some aspects are not included and having the"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Knowing should be tagged as PRESP
        self.assertIn("PRESP", tagged_words[10]["tags"])

    def test_pastp(self):
        text = ". Built in a single week, the house would stand for fifty years"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Built should be tagged as PASTP
        self.assertIn("PASTP", tagged_words[1]["tags"])

    def test_wzpast(self):
        text = (
            "in most cases with understanding and restraint . The progress reported by the advisory "
            "committee is real . While some"
        )
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Reported should be tagged as WZPAST
        self.assertIn("WZPAST", tagged_words[10]["tags"])

        text = "The toy created by the child was innovative. However, the mechanism used in its operation was complex."
        tagged_words = tag_text(text, pipeline=self.pipeline)

        # 'created' should be tagged as WZPAST
        self.assertIn("WZPAST", tagged_words[2]["tags"])

        # Additionally, 'used' should also be tagged as WZPAST
        self.assertIn("WZPAST", tagged_words[13]["tags"])

    def test_wzpres(self):
        text = (
            "and the mean , and he sees the Compson family disintegrating from within . If the barn-burner 's "
            "family produces"
        )
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Disintegrating should be tagged as WZPAST
        self.assertIn("WZPRES", tagged_words[10]["tags"])

    def test_tsub(self):
        text = (
            "we proceed through the seasons of life . Minor characters that surround the love "
            "triangle are colorful and woven with"
        )
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # That should be tagged as a TSUB
        self.assertIn("TSUB", tagged_words[10]["tags"])

    def test_tsub_with_intervening_adverb(self):
        text = "The movies that often showcase historical events are thrilling. It presents a blend of facts and fiction."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # That should be tagged as TSUB
        self.assertIn("TSUB", tagged_words[2]["tags"])

    def test_whsub(self):
        text = (
            ". There are plenty of reference mentioned at the end which can be followed "
            "up for more curiosity . Must"
        )
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Which should be tagged as a WHSUB
        self.assertIn("WHSUB", tagged_words[10]["tags"])

    def test_whobj(self):
        text = "can be brave and courageous . Mafatu is a boy whose mom dies at sea and ever since he was"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Whose should be tagged as a WHOBJ
        self.assertIn("WHOBJ", tagged_words[10]["tags"])

    def test_whobj_complex_sentence(self):
        text = "I have a friend whom everyone in town admires for her kindness. She is truly special."
        tagged_words = tag_text(text, pipeline=self.pipeline)
        self.assertIn("WHOBJ", tagged_words[4]["tags"])

    def test_pire(self):
        text = "pencil ! I am a semi-professional singer , one of whose idols is the great Judy Garland . No one"

        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Whose should be tagged as a PIRE
        self.assertIn("PIRE", tagged_words[12]["tags"])

    def test_sere(self):
        text = "does not stop until you put the book down , which you will not do until you have finished it"

        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Which should be tagged as a SERE
        self.assertIn("SERE", tagged_words[10]["tags"])

    def test_tobj(self):
        text = "the dog that I saw"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # That should be tagged as a TOBJ
        self.assertIn("TOBJ", tagged_words[2]["tags"])

    def test_tobj_with_proper_noun(self):
        text = "The painting that Picasso painted is priceless."
        tagged_words = tag_text(text, pipeline=self.pipeline)

        # That should be tagged as TOBJ
        self.assertIn("TOBJ", tagged_words[2]["tags"])

    def test_caus(self):
        text = "they did also fall under the power of death , because they did eat in disobedience ; and disobedience to"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Because should be tagged as a CAUS
        self.assertIn("CAUS", tagged_words[10]["tags"])

    def test_conc(self):
        text = "outsider . When they learn you 're in the hills though , they 'll rally , do n't worry about"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Though should be tagged as a CONC
        self.assertIn("CONC", tagged_words[10]["tags"])

    def test_cond(self):
        text = "so high that the top falls gently over , as if to show that it really is hair and not"
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # If should be tagged as a COND
        self.assertIn("COND", tagged_words[10]["tags"])

    def test_osub(self):
        text = (
            "his comment on the planter dynasties as they have existed since the decades before the \
        Civil War . It may"
        )
        tagged_words = tag_text(text, pipeline=self.pipeline)
        # Since should be tagged as an OSUB
        self.assertIn("OSUB", tagged_words[10]["tags"])

    def test_causative_subordinators(self):
        """Test causative subordinator 'because'"""
        texts = [
            "I stayed home because I was sick",
            "The experiment failed because of poor controls",
            "Because the weather was bad, we cancelled",
        ]
        for text in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            self.assertTrue(
                any(
                    "CAUS" in word["tags"]
                    for word in tagged_words
                    if word["text"].lower() == "because"
                )
            )

    def test_concessive_subordinators(self):
        """Test concessive subordinators (although, though, tho)"""
        texts = [
            "Although it was raining, we went out",
            "We continued though it was difficult",
            "They succeeded, tho nobody believed in them",
        ]
        for text in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            concessive_words = ["although", "though", "tho"]
            found = False
            for word in tagged_words:
                if word["text"].lower() in concessive_words:
                    self.assertIn("CONC", word["tags"])
                    found = True
            self.assertTrue(found)

    def test_conditional_subordinators(self):
        """Test conditional subordinators (if, unless)"""
        texts = [
            "If it rains, bring an umbrella",
            "Unless you study, you won't pass",
            "The experiment will fail if proper controls aren't used",
        ]
        for text in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            conditional_words = ["if", "unless"]
            found = False
            for word in tagged_words:
                if word["text"].lower() in conditional_words:
                    self.assertIn("COND", word["tags"])
                    found = True
            self.assertTrue(found)

    def test_other_subordinators(self):
        """Test other subordinators"""
        texts = [
            "Since you asked, I'll explain",
            "While the experiment ran, we collected data",
            "Whereas the first trial succeeded, the second failed",
        ]
        for text in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            found = False
            for word in tagged_words:
                if word["text"].lower() in ["since", "while", "whereas"]:
                    self.assertIn("OSUB", word["tags"])
                    found = True
            self.assertTrue(found)

    def test_complex_other_subordinators(self):
        """Test multi-word other subordinators"""
        text = "I'll wait as long as necessary"
        tagged_words = tag_text(text, pipeline=self.pipeline)

        # Find the "as" that starts "as long as"
        for i, word in enumerate(tagged_words):
            if word["text"].lower() == "as" and i + 2 < len(tagged_words):
                if (
                    tagged_words[i + 1]["text"].lower() == "long"
                    and tagged_words[i + 2]["text"].lower() == "as"
                ):
                    self.assertIn("OSUB", word["tags"])
                    return
        self.fail("Multi-word subordinator 'as long as' not properly tagged")

    def test_multiple_subordinators(self):
        """Test text with multiple subordinators"""
        text = "Although it was difficult, we continued because we believed that if we persisted, we would succeed"
        tagged_words = tag_text(text, pipeline=self.pipeline)

        # Find indices dynamically instead of hardcoding
        for i, word in enumerate(tagged_words):
            if word["text"].lower() == "although":
                self.assertIn("CONC", word["tags"])
            elif word["text"].lower() == "because":
                self.assertIn("CAUS", word["tags"])
            elif word["text"].lower() == "if":
                self.assertIn("COND", word["tags"])

    def test_complex_osub_cases(self):
        """Test complex cases of other subordinators"""
        texts = [
            "Insofar as the data shows",
            "Inasmuch as we understand",
            "Such that the results were clear",
            "So that we could proceed",
        ]
        for text in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            # These are multi-word subordinators, need to check first word
            first_word = tagged_words[0]["text"].lower()
            if first_word in ["insofar", "inasmuch", "such", "so"]:
                self.assertIn("OSUB", tagged_words[0]["tags"])

    def test_subordinator_with_punctuation(self):
        """Test subordinators with various punctuation patterns"""
        texts = [
            "We proceeded, although with caution",
            "If, and only if, the conditions are met",
        ]
        expected_tags = {"although": "CONC", "if": "COND"}

        for text in texts:
            tagged_words = tag_text(text, pipeline=self.pipeline)
            found = False
            for word in tagged_words:
                word_lower = word["text"].lower()
                if word_lower in expected_tags:
                    self.assertIn(expected_tags[word_lower], word["tags"])
                    found = True
                    break
            self.assertTrue(found, f"No subordinator found in text: {text}")


if __name__ == "__main__":
    unittest.main()
