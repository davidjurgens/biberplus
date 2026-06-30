import sys
from multiprocessing import Pool

from tqdm import tqdm

sys.path.append('../..')

from biberplus.tagger.function_words_tagger import FunctionWordsTagger
from biberplus.tagger.data_io import simple_split_batching
from biberplus.tagger.tagger_utils import build_variable_dictionaries, load_config, load_pipeline
from biberplus.tagger.biber_plus_tagger import BiberPlusTagger


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

    # No need to batch / parallelize texts below the configured threshold.
    token_count = len(text.split(' '))
    if token_count < config['batching_threshold'] * config['processing_size']:
        return tag_batch(text, config, patterns_dict, pipeline)

    if config['n_processes'] > 1:
        return tag_text_parallel(text, config)

    for text_batch in simple_split_batching(text, config['processing_size'], config['show_progress']):
        all_tagged.extend(tag_batch(text_batch, config, patterns_dict, pipeline))

    return all_tagged


# Per-worker globals: the spaCy pipeline and pattern dictionaries are loaded
# once in each worker process (via the Pool initializer) and reused across all
# batches that worker handles, instead of being reloaded for every batch.
_WORKER_PIPELINE = None
_WORKER_PATTERNS = None
_WORKER_CONFIG = None


def _init_parallel_worker(config):
    global _WORKER_PIPELINE, _WORKER_PATTERNS, _WORKER_CONFIG
    _WORKER_CONFIG = config
    _WORKER_PATTERNS = build_variable_dictionaries()
    _WORKER_PIPELINE = load_pipeline(config)


def _tag_batch_worker(text_batch):
    return tag_batch(text_batch, _WORKER_CONFIG, _WORKER_PATTERNS, _WORKER_PIPELINE)


def tag_text_parallel(text, config):
    # Spawn-safe (Windows/macOS default start method): the worker initializer and
    # batch worker are module-level functions and the initargs are picklable, so
    # no pickling of local closures is required. Callers on those platforms must
    # still guard their entry point with `if __name__ == "__main__":`.
    batches = list(simple_split_batching(text, config['processing_size'], show_progress=False))

    all_tagged = []

    with Pool(config['n_processes'], initializer=_init_parallel_worker, initargs=(config,)) as p:
        # imap preserves input order, so the tagged output stays in document order.
        for tagged_words in tqdm(p.imap(_tag_batch_worker, batches),
                                 total=len(batches), disable=not config['show_progress']):
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
    # Store morphological features as a plain string (e.g. "Tense=Past|VerbForm=Fin")
    # rather than a spaCy MorphAnalysis object. Substring checks like
    # "Tense=Past" in feats behave identically, and unlike MorphAnalysis a string
    # is picklable (required for the multiprocessing path) and JSON-serializable.
    return {'text': word.text,
            'upos': word.pos_,
            'xpos': word.tag_,
            'feats': str(word.morph) if word.morph else "",
            'tags': []}
