import os
from glob import glob

import pandas as pd


def tagged_words_to_tsv(tagged_words, out_file):
    tagged_words_df = pd.DataFrame(tagged_words)
    tagged_words_df.to_csv(out_file, sep='\t', index=False)


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
