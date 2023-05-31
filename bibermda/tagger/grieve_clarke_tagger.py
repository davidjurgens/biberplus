from bibermda.tagger.tag_helper import TagHelper


class GrieveClarkeTagger:
    def __init__(self, tagged_words, patterns_dict):
        self.tagged_words = tagged_words
        self.patterns = patterns_dict
        self.helper = TagHelper(patterns_dict)

    def run_all(self):
        return self.tagged_words
