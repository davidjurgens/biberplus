from math import ceil

from tqdm import tqdm


def simple_text_batching(text, token_batch_size, show_progress=False):
    """ Split on spaces and count the number of tokens. As simple as it gets """
    tokens = text.split(' ')
    iterator = range(0, len(tokens), token_batch_size)
    batch_count = ceil(len(tokens) / token_batch_size)

    if show_progress:
        for i in tqdm(iterator, total=batch_count):
            yield " ".join(tokens[i:i + token_batch_size])
    else:
        for i in range(0, len(tokens), token_batch_size):
            yield " ".join(tokens[i:i + token_batch_size])
