from math import ceil

import spacy
from tqdm import tqdm


def simple_split_batching(text: str, token_batch_size: int, show_progress: bool):
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


def spacy_tokenize_batching(text: str, token_batch_size: int):
    pass


def simple_split_batching_directory():
    pass


def simple_split_batching_lists(text_lists: list, token_batch_size: int):
    pass


def spacy_tokenize_batching_lists(text_lists: list, token_batch_size: int):
    pass


def load_pipeline_for_tokenization(use_gpu):
    if use_gpu:
        spacy.require_gpu()
    else:
        spacy.prefer_gpu()

    return spacy.load("en_core_web_sm", disable=['tagger', 'tok2vec', 'parser', 'lemmatizer', 'ner'])


from glob import glob


def read_directory_of_text_files(dir_path):
    # Ensure directory path ends in a slash
    if dir_path[-1] != '/':
        dir_path += '/'

    txt_paths = glob(dir_path + '*.txt')
    assert len(txt_paths) > 0, "No text files found in the directory!"

    for txt_path in txt_paths:
        yield read_from_file(txt_path)


def read_from_file(txt_path):
    with open(txt_path, 'r') as f:
        return f.read()
