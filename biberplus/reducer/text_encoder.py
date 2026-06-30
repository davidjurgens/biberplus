from biberplus.tagger import calculate_tag_frequencies
from biberplus.tagger.constants import BIBER_PLUS_TAGS, DOC_TAGS

# Columns produced by calculate_tag_frequencies (order matters: it determines the
# layout of the flattened vectors). Biber/function-word groups use all of them;
# the binary group uses only mean/std.
VALUE_COLUMNS = ['mean', 'min_val', 'max_val', 'range', 'std']
BINARY_COLUMNS = ['mean', 'std']


def _flatten_group(df_subset, columns, round_to):
    """Flatten a frequencies subset to a (names, values) pair.

    The flatten is row-major, so a tag's columns are contiguous and each value at
    position i is described by names[i] (e.g. 'PASS_mean'). Names are derived from
    the actual rows, so they stay aligned regardless of tag ordering.
    """
    values = df_subset[columns].to_numpy().flatten().round(round_to).tolist()
    names = [f"{tag}_{col}" for tag in df_subset['tag'] for col in columns]
    return names, values


def encode_frequencies(frequencies_df, config, round_to=10):
    """Encode a precomputed frequencies DataFrame into labeled vectors.

    Returns a dict keyed by group ('biber', and optionally 'binary' and
    'function_words'); each value is {'names': [...], 'values': [...]} where names
    and values are positionally aligned.
    """
    encodings = {}

    biber_tags = BIBER_PLUS_TAGS + DOC_TAGS
    binary_tags = ['BIN_' + tag for tag in biber_tags]

    if config['binary_tags']:
        subset = frequencies_df[frequencies_df['tag'].isin(binary_tags)]
        names, values = _flatten_group(subset, BINARY_COLUMNS, round_to)
        encodings['binary'] = {'names': names, 'values': values}

    if config['function_words']:
        subset = frequencies_df[~frequencies_df['tag'].isin(biber_tags + binary_tags)]
        names, values = _flatten_group(subset, VALUE_COLUMNS, round_to)
        encodings['function_words'] = {'names': names, 'values': values}

    subset = frequencies_df[frequencies_df['tag'].isin(biber_tags)]
    names, values = _flatten_group(subset, VALUE_COLUMNS, round_to)
    encodings['biber'] = {'names': names, 'values': values}

    return encodings


def encode_text(config, text, round_to=10):
    """Tag `text` and return labeled encoding vectors.

    Each group is a dict {'names': [...], 'values': [...]} so callers get the
    feature name for every position, not just the raw vector. Use values-only via
    `encodings['biber']['values']`.
    """
    frequencies_df = calculate_tag_frequencies(text, config=config)
    return encode_frequencies(frequencies_df, config, round_to)
