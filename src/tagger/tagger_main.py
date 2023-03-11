import argparse
import sys

import pandas as pd
import stanza

sys.path.append('../..')

from src.tagger.data_io import simple_text_batching
from src.tagger.tagger_utils import build_variable_dictionaries, tagged_words_to_tsv
from src.tagger.word_tagger import WordTagger


def run_tagger(pipeline, text, token_batch_size, show_progress=False):
    patterns_dict = build_variable_dictionaries()

    for text_batch in simple_text_batching(text, token_batch_size, show_progress):
        doc = pipeline(text_batch)
        word_tagger = WordTagger(doc, patterns_dict)
        word_tagger.run_all()
        yield pd.DataFrame(word_tagger.tagged_words)


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
