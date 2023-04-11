from .corpus_statistics import calculate_corpus_statistics, calculate_corpus_statistics_parallel
from ..tagger import tag_string, tag_large_string, tag_string_parallel

__all__ = ['calculate_corpus_statistics', 'calculate_corpus_statistics_parallel', 'tag_string', 'tag_large_string',
           'tag_string_parallel']
