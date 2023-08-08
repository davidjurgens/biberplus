import sys
from multiprocessing import Pool

from tqdm import tqdm

sys.path.append('../..')

from bibermda.tagger.function_words_tagger import FunctionWordsTagger
from bibermda.tagger.data_io import simple_split_batching
from bibermda.tagger.tagger_utils import build_variable_dictionaries, load_config, load_pipeline
from bibermda.tagger.biber_plus_tagger import BiberPlusTagger


def tag_text(text, pipeline=None, config=None):
    """
    :param text: The text to tag
    :param pipeline: Spacy pipeline
    :param config: The settings/parameters for the tagging
    :return: List of tagged words where each word is a dictionary of values
    """
    config = config or load_config()
    pipeline = pipeline or load_pipeline(config)
    patterns_dict = build_variable_dictionaries()
    all_tagged = []

    # No need to batch / parallelize small texts
    if len(text.split(' ')) < config['processing_size'] * 10:  # Arbitrary cutoff
        return tag_batch(text, config, patterns_dict, pipeline)

    if config['n_processes'] > 1:
        return tag_text_parallel(text, config)

    for text_batch in simple_split_batching(text, config['processing_size'], config['show_progress']):
        all_tagged.extend(tag_batch(text_batch, config, patterns_dict, pipeline))

    return all_tagged


def tag_text_parallel(text, config):
    patterns_dict = build_variable_dictionaries()

    # Split the text into batches
    process_args = []

    for text_batch in simple_split_batching(text, config['processing_size'], show_progress=False):
        process_args.append((text_batch, config, patterns_dict, None))

    all_tagged = []

    with Pool(config['n_processes']) as p:
        for tagged_words in p.starmap(tag_batch,
                                      tqdm(process_args, total=len(process_args), disable=not config['show_progress'])):
            all_tagged.extend(tagged_words)

    return all_tagged


def tag_batch(text_batch, config, patterns_dict, pipeline=None):
    """Tag a batch of text."""
    pipeline = pipeline or load_pipeline(config)
    doc = pipeline(text_batch)
    tagged_words = [word2dict(word) for word in doc]
    tagged_words = tag_function_words(tagged_words, config)
    tagged_words = tag_biber_and_binary(tagged_words, patterns_dict, config)
    return tagged_words


def tag_function_words(tagged_words, config):
    if config['function_words']:
        return FunctionWordsTagger(tagged_words, config['function_words_list']).tag()
    return tagged_words


def tag_biber_and_binary(tagged_words, patterns_dict, config):
    if config['biber'] or config['binary_tags']:
        return BiberPlusTagger(tagged_words, patterns_dict).run_all()
    return tagged_words


def word2dict(word):
    return {'text': word.text,
            'upos': word.pos_,
            'xpos': word.tag_,
            'feats': word.morph if word.morph else "",
            'tags': []}
