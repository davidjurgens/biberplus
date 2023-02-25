import argparse

import stanza

from tagger_utils import build_variable_dictionaries, tagged_words_to_tsv
from word_tagger import WordTagger


def run_tagger(nlp_pipeline, text):
    patterns_dict = build_variable_dictionaries()
    doc = nlp_pipeline(text)

    word_tagger = WordTagger(doc, patterns_dict)
    word_tagger.run_all()

    return word_tagger.tagged_words


def tag_text(text, use_gpu=True):
    patterns_dict = build_variable_dictionaries()
    pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=use_gpu)
    doc = pipeline(text)

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

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text_file", type=str)
    group.add_argument("--input_directory", type=str)

    parser.add_argument("--output_all_tags", default=False, type=str, action='store_true')
    parser.add_argument("--use_gpu", default=False, action='store_true', type=bool, required=False)


    args = parser.parse_args()

    main(args)
