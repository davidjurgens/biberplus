import re

from bibermda.tagger.tag_helper import TagHelper


class GrieveTagger:
    def __init__(self, tagged_sentence, patterns_dict):
        self.tagged_sentence = tagged_sentence
        self.text = " ".join(tagged_word['text'] for tagged_word in tagged_sentence)
        self.helper = TagHelper(patterns_dict)

    def contains_question_mark(self):
        pattern = r'\?'
        return re.search(pattern, self.text, flags=re.IGNORECASE)

    def contains_quotation_mark(self):
        pattern = r'"'
        return re.search(pattern, self.text, flags=re.IGNORECASE)

    def contains_url(self):
        pattern = "https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"
        return re.search(pattern, self.text, flags=re.IGNORECASE)

    def contains_numerals(self):
        return self.helper.is_numeral()

    def contains_quantifiers(self):
        pattern = re.compile(
            r'\b(?:all|some|many|few|several|most|fewer|more|plenty of|lots of|a lot of|a few|a couple of)\b',
            re.IGNORECASE)
        return re.search(pattern, self.text, flags=re.IGNORECASE)

    def contains_hashtag(self):
        pass

    def contains_capitalisations(self):
        pass

    def contains_emoticons(self):
        pass

    def contains_exclamation_marks(self):
        pass
