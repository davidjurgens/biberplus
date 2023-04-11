import sys

import numpy as np
import pandas as pd
import spacy

sys.path.append('../..')

from bibermda.tagger.data_io import simple_split_batching
from bibermda.tagger.tagger_utils import build_variable_dictionaries
from bibermda.tagger.word_tagger import WordTagger
from math import ceil


def tag_large_string(pipeline, text, out_tsv, token_batch_size=1000, show_progress=False):
    patterns_dict = build_variable_dictionaries()
    all_tagged = []

    for text_batch in simple_split_batching(text, token_batch_size, show_progress):
        doc = pipeline(text_batch)
        word_tagger = WordTagger(words=list(doc), patterns_dict=patterns_dict)
        word_tagger.run_all()
        all_tagged.extend(word_tagger.tagged_words)

    df = pd.DataFrame(all_tagged)
    df.to_csv(out_tsv, sep='\t', index=False, compression='gzip')


def tag_string_batched(pipeline, text, token_batch_size=1000, n_cpu=1, show_progress=False):
    patterns_dict = build_variable_dictionaries()

    for text_batch in simple_split_batching(text, token_batch_size, show_progress):
        if n_cpu == 1:
            doc = pipeline(text_batch)
            word_tagger = WordTagger(words=list(doc), patterns_dict=patterns_dict)
        else:
            # TODO: Test further before adding to the documentation
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


def tag_string(pipeline, text):
    patterns_dict = build_variable_dictionaries()
    doc = pipeline(text)
    word_tagger = WordTagger(words=list(doc), patterns_dict=patterns_dict)
    word_tagger.run_all()
    return word_tagger.tagged_words


def load_pipeline(use_gpu):
    if use_gpu:
        spacy.require_gpu()
    else:
        spacy.prefer_gpu()

    return spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])
