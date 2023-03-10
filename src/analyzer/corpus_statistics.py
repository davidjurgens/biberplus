import functools
import operator
import sys

import numpy as np
import pandas as pd

sys.path.append('../..')

from collections import defaultdict
from src.analyzer.constants import all_tags
from src.tagger.tagger_main import run_tagger
from math import ceil


def calculate_corpus_statistics(text, token_batch_size, token_normalization=100, use_gpu=True, show_progress=False):
    corpus_tag_counts = defaultdict(list, {k: [] for k in all_tags})

    for tagged_dataframe in run_tagger(text, token_batch_size, use_gpu, show_progress):
        corpus_tag_counts = count_tags_every_n_tokens(tagged_dataframe, corpus_tag_counts, n_tokens=token_normalization)

    return calculate_descriptive_stats(corpus_tag_counts)


def calculate_descriptive_stats(tag_counts):
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


def count_tags_every_n_tokens(tagged_df, tag_counts, n_tokens):
    num_batches = ceil(len(tagged_df) / n_tokens)

    for index, batch in enumerate(np.array_split(tagged_df, num_batches)):
        tag_counts = update_tag_counts(tagged_df=batch, tag_counts=tag_counts)
        # Account for last batch which may be smaller than n_tokens
        if index == num_batches - 1:
            weight = len(batch) / n_tokens
            for k, v in tag_counts.items():
                v[index] *= weight

    return tag_counts


def update_tag_counts(tagged_df, tag_counts):
    curr_counts = pd.Series(functools.reduce(operator.iconcat, tagged_df.tags, [])).value_counts().to_dict()

    # TODO: Speed this up?
    for tag in all_tags:
        if tag in curr_counts:
            tag_counts[tag].append(int(curr_counts[tag]))
        else:
            tag_counts[tag].append(0)

    return tag_counts
