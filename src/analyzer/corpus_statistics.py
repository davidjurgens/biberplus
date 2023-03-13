import functools
import operator
import sys

import numpy as np
import pandas as pd
import spacy

sys.path.append('../..')

from collections import defaultdict
from src.analyzer.constants import ALL_TAGS, DOC_TAGS
from src.tagger.tagger_main import run_tagger
from math import ceil


def calculate_corpus_statistics(text, token_batch_size, token_normalization=1000,
                                use_gpu=True, n_cpu=1, show_progress=False, pipeline=None):
    if not pipeline:
        pipeline = load_pipeline(use_gpu)

    corpus_tag_counts = defaultdict(list, {k: [] for k in ALL_TAGS})
    corpus_doc_counts = defaultdict(list, {k: [] for k in DOC_TAGS})

    for word_tagger in run_tagger(pipeline, text, token_batch_size, n_cpu, show_progress):
        tagged_dataframe = pd.DataFrame(word_tagger.tagged_words)
        corpus_tag_counts, corpus_doc_counts = count_tags_every_n_tokens(tagged_dataframe, corpus_tag_counts,
                                                                         corpus_doc_counts, token_normalization)

    return calculate_descriptive_stats(corpus_tag_counts, corpus_doc_counts)


def calculate_descriptive_stats(tag_counts, doc_counts):
    tag_counts = dict(list(tag_counts.items()) + list(doc_counts.items()))
    rows = []

    for tag, counts in tag_counts.items():
        counts = np.array(counts)
        rows.append({
            'tag': tag,
            'mean': counts.mean(),
            'min_val': min(counts),
            'max_val': max(counts),
            'range': np.ptp(counts),
            'std': counts.std()
        })

    return pd.DataFrame(rows)


def count_tags_every_n_tokens(tagged_df, tag_counts, doc_counts, n_tokens):
    num_batches = ceil(len(tagged_df) / n_tokens)

    for index, batch in enumerate(np.array_split(tagged_df, num_batches)):
        tag_counts = update_tag_counts(tagged_df=batch, tag_counts=tag_counts)
        doc_counts = update_doc_level_counts(tagged_df=batch, doc_counts=doc_counts)
        # Account for last batch which may be smaller than n_tokens
        if index == num_batches - 1:
            weight = len(batch) / n_tokens
            for k, v in tag_counts.items():
                v[index] *= weight

    return tag_counts, doc_counts


def update_tag_counts(tagged_df, tag_counts):
    curr_counts = pd.Series(functools.reduce(operator.iconcat, tagged_df.tags, []),
                            dtype=pd.StringDtype()).value_counts().to_dict()

    # TODO: Speed this up?
    for tag in ALL_TAGS:
        if tag in curr_counts:
            tag_counts[tag].append(int(curr_counts[tag]))
        else:
            tag_counts[tag].append(0)

    return tag_counts


def update_doc_level_counts(tagged_df, doc_counts):
    doc_counts['AWL'].append(calculate_mean_word_length(tagged_df))
    doc_counts['RB'].append(calculate_total_adverbs(tagged_df))
    doc_counts['TTR'].append(calculate_type_token_ratio(tagged_df))
    return doc_counts


def calculate_total_adverbs(tagged_df):
    return len(tagged_df[tagged_df['upos'] == 'ADV'])


def calculate_mean_word_length(tagged_df):
    return tagged_df['text'].apply(len).mean()


def calculate_type_token_ratio(tagged_df, first_n=400):
    if first_n:
        tagged_df = tagged_df.iloc[:first_n]

    uniq_vocab = set(tagged_df['text'].unique())

    return len(uniq_vocab) / len(tagged_df)


def load_pipeline(use_gpu):
    if use_gpu:
        spacy.require_gpu()
    else:
        spacy.prefer_gpu()

    return spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner'])
