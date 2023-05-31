import os
from glob import glob

import spacy
import yaml


def load_config():
    config_fp = os.path.dirname(__file__) + '/config.yaml'
    with open(config_fp) as f:
        return yaml.safe_load(f)


def build_variable_dictionaries():
    script_dir = os.path.dirname(__file__)
    constant_files = glob(script_dir + '/constants/*.txt')
    variables_dict = {}

    for constant_file in constant_files:
        # E.g. constants/suasive_verbs.txt -> suasive_verbs
        file_name = constant_file.split('/')[-1].replace('.txt', '')
        variables_dict[file_name] = read_in_variables(constant_file)

    return variables_dict


def read_in_variables(txt_file):
    variables = []
    with open(txt_file, 'r') as f:
        for line in f:
            var = line.strip()
            if var:
                variables.append(var)
    return set(variables)


def load_pipeline(config):
    if config['use_gpu']:
        spacy.require_gpu()
    else:
        spacy.prefer_gpu()

    return spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer', 'ner', 'textcat'])


def load_tokenizer(use_gpu=False):
    if use_gpu:
        spacy.require_gpu()
    else:
        spacy.prefer_gpu()

    return spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'lemmatizer', 'ner', 'textcat'])
