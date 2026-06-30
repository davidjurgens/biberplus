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
