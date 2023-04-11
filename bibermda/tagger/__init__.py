from .tagger import load_pipeline, tag_string, tag_large_string
from .tagger_parallel import tag_string_parallel

__all__ = ['load_pipeline', 'tag_string', 'tag_large_string', 'tag_string_parallel']