"""Which Biber features fire on each token? (rule tagger — no torch needed)

    python examples/tag_text.py
"""
from biberplus.tagger.biber_plus_tagger import BiberPlusTagger
from biberplus.tagger.tagger import word2dict
from biberplus.tagger.tagger_utils import (build_variable_dictionaries,
                                           load_config, load_pipeline)
from biberplus.neurobiber.features import name_of, structural_codes

TEXT = "It doesn't seem likely that we'll finish today, but I really hope we do!"

config = load_config()
config["function_words"] = False
doc = load_pipeline(config)(TEXT)
tagged = BiberPlusTagger([word2dict(t) for t in doc],
                         build_variable_dictionaries()).run_all()

skip = structural_codes()  # hide nouns/articles/prepositions noise
for word in tagged:
    tags = [t for t in word["tags"] if t not in skip]
    if tags:
        print(f"{word['text']:12s} {', '.join(f'{t} ({name_of(t)})' for t in tags)}")
