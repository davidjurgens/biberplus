import functools
import operator
from collections import defaultdict
from math import ceil

import numpy as np
import pandas as pd

from bibermda.tagger import tag_text
from bibermda.tagger.constants import BIBER_TAGS, GRIEVE_CLARK_TAGS
from bibermda.tagger.tagger_utils import load_config, load_pipeline, build_variable_dictionaries


def calculate_tag_frequencies(text, pipeline=None, config=None):
    config = config or load_config()
    pipeline = pipeline or load_pipeline(config)
    tags = load_tags(config)

    tag_frequencies = defaultdict(list)
    tagged_words = tag_text(text, pipeline, config)
    tagged_dataframe = pd.DataFrame(tagged_words)
    tag_frequencies = count_tags_every_n_tokens(tagged_dataframe, tag_frequencies, tags, config)

    return calculate_descriptive_stats(tag_frequencies)


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
        if index != num_batches - 1:
            tag_counts = update_tag_counts(batch, tag_counts, tags=tags)
        else:
            # Ignore the last batch if it's too small, otherwise scale up tag frequencies
            percent = len(batch) / config['token_normalization']
            if percent > config['drop_last_batch_pct']:
                tag_counts = update_tag_counts(batch, tag_counts, tags,
                                               weight=config['token_normalization'] / len(batch))

    return tag_counts


def update_tag_counts(tagged_df, tag_counts, tags, weight=1.):
    curr_counts = pd.Series(functools.reduce(operator.iconcat, tagged_df.tags, []),
                            dtype=pd.StringDtype()).value_counts().to_dict()

    for tag in tags:
        if tag in curr_counts:
            tag_counts[tag].append(round(curr_counts[tag] * weight))
        else:
            tag_counts[tag].append(0)

    # Update document level tags
    tag_counts['AWL'].append(calculate_mean_word_length(tagged_df))
    tag_counts['RB'].append(calculate_total_adverbs(tagged_df))
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
        tags.extend(BIBER_TAGS)
    if config['grieve_clarke']:
        tags.extend(GRIEVE_CLARK_TAGS)

    if config['function_words']:
        fw = config['function_words_list'] if config['function_words_list'] else build_variable_dictionaries()[
            'function_words']
        tags.extend(fw)

    return tags
