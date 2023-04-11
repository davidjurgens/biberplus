import sys

import pandas as pd
import spacy

sys.path.append('../..')

from multiprocessing import Pool
import multiprocessing
from bibermda.tagger.data_io import simple_split_batching

from bibermda.tagger.tagger_utils import build_variable_dictionaries
from bibermda.tagger.word_tagger import WordTagger
from tqdm import tqdm


def tag_string_parallel(text, out_tsv, token_batch_size=10000, n_processes=4, use_gpu=True, show_progress=False):
    patterns_dict = build_variable_dictionaries()

    # Split the text into batches
    process_args = []

    for text_batch in simple_split_batching(text, token_batch_size, show_progress=False):
        process_args.append((patterns_dict, text_batch, use_gpu))

    all_tagged = []

    with Pool(n_processes) as p:
        # TODO: Avoid repeating code
        if show_progress:
            for tagged_words in p.starmap(tag_string, tqdm(process_args, total=len(process_args))):
                all_tagged.extend(tagged_words)
        else:
            for tagged_words in p.starmap(tag_string, process_args):
                all_tagged.extend(tagged_words)

    df = pd.DataFrame(all_tagged)
    df.to_csv(out_tsv, sep='\t', index=False, compression='gzip')


def tag_string(patterns_dict, text, use_gpu):
    # Throws an error when I pass this in the starmap args.
    # TODO: Can we avoid instantiating this every time
    pipeline = load_pipeline(use_gpu)
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


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    print("Reading in pubmed data")
    pubmed_fp = '../../data/pubmed.txt'

    pipeline = load_pipeline(use_gpu=True)

    with open(pubmed_fp, 'r', encoding='utf-8') as f:
        text = f.read()

    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    print(len(text.split(' ')))
    sample = " ".join(text.split(' ')[:500000])
    for tagged_words in tag_string_batched(text, token_batch_size=50000, n_processes=12, use_gpu=True,
                                           show_progress=True):
        print(len(tagged_words))
        df = pd.DataFrame(tagged_words)
