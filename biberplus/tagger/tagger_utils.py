import copy
import os
from functools import lru_cache
from pathlib import Path

import spacy
import yaml


@lru_cache(maxsize=1)
def _load_config_cached():
    config_fp = Path(os.path.dirname(__file__)) / 'config.yaml'
    with open(config_fp) as f:
        return yaml.safe_load(f)


def load_config():
    # Return a deep copy so callers can freely mutate (e.g. config.update(...))
    # without corrupting the cached parse shared across calls.
    return copy.deepcopy(_load_config_cached())


@lru_cache(maxsize=1)
def build_variable_dictionaries():
    script_dir = Path(os.path.dirname(__file__))
    constant_files = script_dir.glob('constants/*.txt')
    variables_dict = {}

    for constant_file in constant_files:
        # E.g. constants/suasive_verbs.txt -> suasive_verbs
        file_name = constant_file.stem
        variables_dict[file_name] = read_in_variables(constant_file)

    return variables_dict


def read_in_variables(txt_file):
    variables = []
    # Explicit UTF-8 so lexicon files load identically on Windows (whose default
    # encoding is cp1252) as on Linux/macOS.
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            var = line.strip()
            if var:
                variables.append(var)
    return set(variables)


def load_pipeline(config):
    if config['use_gpu']:
        spacy.require_gpu()

    return spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner', 'textcat'])


def load_tokenizer(use_gpu=False):
    if use_gpu:
        spacy.require_gpu()

    return spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'lemmatizer', 'ner', 'textcat'])
