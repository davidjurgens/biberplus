import argparse

import stanza

from word_tagger import WordTagger
from tagger_utils import build_variable_dictionaries, tagged_words_to_tsv
from word_tagger import WordTagger


def run_tagger(nlp_pipeline, text):
    patterns_dict = build_variable_dictionaries()
    doc = nlp_pipeline(text)

    word_tagger = WordTagger(doc, patterns_dict)
    word_tagger.run_all()

    return word_tagger.tagged_words


def main(args):
    pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=args.use_gpu)

    with open(args.text_file, 'r') as f:
        text = f.read()

    tagged_sents = run_tagger(pipeline, text)

    if args.output_file:
        tagged_words_to_tsv(tagged_sents, args.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--text_file", type=str, required=True)
    parser.add_argument("--use_gpu", default=False, action='store_true', type=bool, required=False)
    parser.add_argument("--output_file", default=None, type=str, required=False)

    args = parser.parse_args()

    main(args)
