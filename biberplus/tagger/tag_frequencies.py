import warnings
import functools
import operator
from collections import defaultdict
from math import ceil

import numpy as np
import pandas as pd

from biberplus.tagger import tag_text
from biberplus.tagger.constants import BIBER_PLUS_TAGS
from biberplus.tagger.tagger_utils import load_config, load_pipeline, build_variable_dictionaries


warnings.filterwarnings('ignore', category=FutureWarning, message='.*swapaxes.*')


def calculate_tag_frequencies(text, pipeline=None, config=None):
    config = config or load_config()
    pipeline = pipeline or load_pipeline(config)
    tags = load_tags(config)

    tag_frequencies = defaultdict(list)
    try:
        tagged_words = tag_text(text, pipeline, config)
        tagged_dataframe = pd.DataFrame(tagged_words)
        tag_frequencies = count_tags_every_n_tokens(tagged_dataframe, tag_frequencies, tags, config)

        return calculate_descriptive_stats(tag_frequencies)
    except Exception as e:
        print(text)
        print(e)

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


def count_tags_every_n_tokens(tagged_df, tag_counts, tags, config):
    num_batches = ceil(len(tagged_df) / config['token_normalization'])

    for index, batch in enumerate(np.array_split(tagged_df, num_batches)):
        last_batch = index == num_batches - 1
        # Ignore the last batch if it's too small, otherwise scale up tag frequencies
        if last_batch and len(batch) <= config['drop_last_batch_pct'] * config['token_normalization']:
            break

        weight = config['token_normalization'] / len(batch) if last_batch else 1.0
        tag_counts = update_tag_counts(batch, tag_counts, tags, tag_binary=config['binary_tags'], weight=weight)

    return tag_counts


def update_tag_counts(tagged_df, tag_counts, tags, tag_binary, weight=1.):
    curr_counts = pd.Series(functools.reduce(operator.iconcat, tagged_df.tags, []),
                            dtype=pd.StringDtype()).value_counts().to_dict()

    for tag in tags:
        count = round(curr_counts[tag] * weight) if tag in curr_counts else 0
        tag_counts[tag].append(count)

        if tag_binary and tag in BIBER_PLUS_TAGS:
            tag_name = 'BIN_' + tag if tag[:4] != 'BIN_' else tag
            tag_counts[tag_name].append(int(tag in curr_counts))

    # Update document level tags
    tag_counts['AWL'].append(calculate_mean_word_length(tagged_df))
    # tag_counts['RB'].append(calculate_total_adverbs(tagged_df))
    tag_counts['TTR'].append(calculate_type_token_ratio(tagged_df))

    return tag_counts


def calculate_total_adverbs(tagged_df):
    return len(tagged_df[tagged_df['upos'] == 'ADV'])


def calculate_mean_word_length(tagged_df):
    return tagged_df['text'].apply(len).mean()


def calculate_type_token_ratio(tagged_df, first_n=400):
    if first_n:
        tagged_df = tagged_df.iloc[:first_n]

    uniq_vocab = set(tagged_df['text'].unique())

    return len(uniq_vocab) / len(tagged_df)


def load_tags(config):
    tags = []
    if config['biber']:
        tags.extend(BIBER_PLUS_TAGS)
    if config['binary_tags']:
        binary_tags = ['BIN_' + tag for tag in BIBER_PLUS_TAGS]
        tags.extend(binary_tags)

    if config['function_words']:
        fw = config['function_words_list'] if config['function_words_list'] else build_variable_dictionaries()[
            'function_words']
        tags.extend(fw)

    return tags


def get_tagged_words(text, pipeline=None, config=None):
    config = config or load_config()
    pipeline = pipeline or load_pipeline(config)
    tagged_words = tag_text(text, pipeline, config)
    return tagged_words
