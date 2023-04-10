import argparse
import sys

import numpy as np
import pandas as pd
import spacy

sys.path.append('../..')

from multiprocessing import Pool
from src.tagger.data_io import simple_split_batching
from src.tagger.tagger_utils import build_variable_dictionaries, tagged_words_to_tsv
from src.tagger.word_tagger import WordTagger
from math import ceil

results = []


def tag_string_batched_parallel(pipeline, text, token_batch_size=1000, n_processes=1, show_progress=False):
    patterns_dict = build_variable_dictionaries()

    # Split the text into batches
    text_batches = list(simple_split_batching(text, token_batch_size, show_progress=False))

    if n_processes > 1:
        with Pool(n_processes) as p:
            for text_batch in text_batches:
                p.map(t, [1, 2, 3])

    # Conver
    for text_batch in simple_split_batching(text, token_batch_size, show_progress):
        if n_processes > 1:
            pool = Pool(n_processes)
        if n_cpu == 1:
            doc = pipeline(text_batch)
            word_tagger = WordTagger(words=list(doc), patterns_dict=patterns_dict)
        else:
            words = []
            # Split the batch and run on multiple CPUS
            tokens = text_batch.split(' ')
            num_batches = ceil(len(tokens) / n_cpu)
            token_batches = np.array_split(tokens, num_batches)
            split_texts = [" ".join(batch) for batch in token_batches]

            for doc in pipeline.pipe(split_texts, n_process=n_cpu, batch_size=500):
                words.extend(list(doc))
            word_tagger = WordTagger(words, patterns_dict)

        word_tagger.run_all()
        yield word_tagger


def tag_string_batched(pipeline, text, token_batch_size=1000, n_cpu=1, show_progress=False):
    patterns_dict = build_variable_dictionaries()

    # Split the text into batches
    text_batches = list(simple_split_batching(text, token_batch_size, show_progress=False))

    # Conver
    for text_batch in simple_split_batching(text, token_batch_size, show_progress):
        if n_cpu == 1:
            doc = pipeline(text_batch)
            word_tagger = WordTagger(words=list(doc), patterns_dict=patterns_dict)
        else:
            words = []
            # Split the batch and run on multiple CPUS
            tokens = text_batch.split(' ')
            num_batches = ceil(len(tokens) / n_cpu)
            token_batches = np.array_split(tokens, num_batches)
            split_texts = [" ".join(batch) for batch in token_batches]

            for doc in pipeline.pipe(split_texts, n_process=n_cpu, batch_size=500):
                words.extend(list(doc))
            word_tagger = WordTagger(words, patterns_dict)

        word_tagger.run_all()
        yield word_tagger


def tag_text(text, use_gpu=True):
    pipeline = load_pipeline(use_gpu)
    return tag_string(pipeline, text)


def tag_string(pipeline, text):
    patterns_dict = build_variable_dictionaries()
    doc = pipeline(text)
    word_tagger = WordTagger(words=list(doc), patterns_dict=patterns_dict)
    word_tagger.run_all()
    return word_tagger.tagged_words


def callback(tagged_words):
    tagged_dataframe = pd.DataFrame(tagged_words)
    results.append(tagged_dataframe)


def main(args):
    pipeline = load_pipeline(args.use_gpu)

    with open(args.text_file, 'r') as f:
        text = f.read()

    tagged_sents = tag_string_batched(pipeline, text, token_batch_size=100, show_progress=False)

    if args.output_file:
        tagged_words_to_tsv(tagged_sents, args.output_file)


def load_pipeline(use_gpu):
    if use_gpu:
        spacy.require_gpu()
    else:
        spacy.prefer_gpu()

    return spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'lemmatizer', 'ner'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text_file", type=str)
    group.add_argument("--input_directory", type=str)

    parser.add_argument("--output_all_tags", default=False, type=str, action='store_true')
    parser.add_argument("--use_gpu", default=False, action='store_true', type=bool, required=False)

    args = parser.parse_args()

    main(args)
