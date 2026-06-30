from math import ceil
from pathlib import Path

from tqdm import tqdm


def simple_split_batching(text: str, token_batch_size: int, show_progress: bool):
    """ Split on spaces and count the number of tokens. As simple as it gets """
    tokens = text.split(' ')
    iterator = range(0, len(tokens), token_batch_size)
    batch_count = ceil(len(tokens) / token_batch_size)

    for i in tqdm(iterator, total=batch_count, disable=not show_progress):
        yield " ".join(tokens[i:i + token_batch_size])


def read_directory_of_text_files(dir_path):
    # Use pathlib so path joining and globbing work cross-platform (no manual
    # '/' concatenation, which is brittle on Windows).
    txt_paths = sorted(Path(dir_path).glob('*.txt'))
    assert len(txt_paths) > 0, "No text files found in the directory!"

    for txt_path in txt_paths:
        yield read_from_file(txt_path)


def read_from_file(txt_path):
    # Explicit UTF-8 so corpora read consistently across platforms.
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()
