from collections import Counter

from biberplus.tagger.tagger_utils import build_variable_dictionaries


class FunctionWordsTagger:
    def __init__(self, tagged_words, function_words):
        self.tagged_words = tagged_words
        if function_words:
            self.function_words = set([w.lower().strip() for w in function_words])
        else:
            self.function_words = build_variable_dictionaries()['function_words']

    def tag(self):
        for word in self.tagged_words:
            if word['text'].lower() in self.function_words:
                word['tags'].append(word['text'].lower())
        return self.tagged_words

    # def is_function_word(self, word):
    #     return self.helper.is_adposition(word) or \
    #            self.helper.is_auxiliary(word) or \
    #            self.helper.is_coordinating_conjunction(word) or \
    #            self.helper.is_determiner(word) or \
    #            self.helper.is_particle(word) or \
    #            self.helper.is_pronoun(word) or \
    #            self.helper.is_subordinating_conjunction(word)


def tag_function_words(words, function_words=None, return_vector=False):
    if function_words:
        function_words = set([w.lower().strip() for w in function_words])
    else:
        function_words = build_variable_dictionaries()['function_words']

    function_word_counts = Counter(word.lower() for word in words if word.lower() in function_words)

    # Sort into the original order of the function words variable
    if return_vector:
        return [function_word_counts[word] for word in function_words]
    return function_word_counts
