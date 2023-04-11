__version__ = "0.1.0"
__author__ = 'Kenan Alkiek'
__credits__ = 'University of Michigan - The Blablablab'

from .analyzer import calculate_corpus_statistics, calculate_corpus_statistics_parallel
from .tagger import load_pipeline, tag_string, tag_large_string
from .tagger import tag_string_parallel
