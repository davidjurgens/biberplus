import argparse

import stanza

from simple_word_tagger import SimpleWordTagger
from tagger_utils import build_variable_dictionaries, save_tagged_doc
from word_tagger import WordTagger


def run_tagger(nlp_pipeline, text):
    patterns_dict = build_variable_dictionaries()
    doc = nlp_pipeline(text)
    doc = doc.to_dict()
    tagged_sentences = []

    for sent_index, sentence in enumerate(doc):
        tagged_sentence = []
        for word_index, word in enumerate(sentence):
            # Tags everything that does have any prior dependencies (2/3rds of the tags have no dependencies)
            simple_tagger = SimpleWordTagger(sentence, word, word_index, patterns_dict)
            simple_tagger.run_all()
            word = simple_tagger.word
            # Tags everything that requires a tag before it to be complete. Requires specific order
            word_tagger = WordTagger(sentence, word, word_index, patterns_dict)
            word_tagger.run_all()
            tagged_sentence.append(word_tagger.word)

        tagged_sentences.append(tagged_sentence)

    return tagged_sentences


def main(args):
    pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=args.use_gpu)

    with open(args.text_file, 'r') as f:
        text = f.read()

    tagged_sents = run_tagger(pipeline, text)

    if args.output_file:
        save_tagged_doc(tagged_sents)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--text_file", type=str, required=True)
    parser.add_argument("--use_gpu", default=False, action='store_true', type=bool, required=False)
    parser.add_argument("--output_file", default=None, type=str, required=False)

    args = parser.parse_args()

    main(args)
