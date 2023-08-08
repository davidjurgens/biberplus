import json
import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def partition(input_file, output_path, nrows=None):
    df = load_metadata(input_file, nrows)
    samples_X, samples_Y = sample_text_pairs(df)
    partition = split_train_dev_test(samples_X['authorID'].tolist())
    save_text_samples(samples_X, samples_Y, partition, input_file, output_path)


def load_metadata(path, nrows=None):
    logging.info("Loading meta data...")
    df = pd.read_json(path, orient='records', lines=True, nrows=nrows)
    df['authorID'] = df['authorIDs'].astype('str')
    return df


def sample_text_pairs(df):
    print("Sampling text pairs...")
    samples_X = df.groupby(['authorID']).sample(1)
    samples_Y = df[~df['documentID'].isin(samples_X['documentID'])].groupby(['authorID']).sample(1)
    del df

    if len(samples_X) > len(samples_Y):
        samples_X = samples_X[samples_X['authorID'].isin(samples_Y['authorID'])]

    if len(samples_Y) < len(samples_X):
        samples_Y = samples_Y[samples_Y['authorID'].isin(samples_X['authorID'])]

    assert set(samples_Y['documentID'].tolist()).intersection(set(samples_X['documentID'].tolist())) == set()
    assert len(samples_X['authorID'].sort_values().tolist()) == len(samples_Y['authorID'].sort_values().tolist())

    return samples_X, samples_Y


def split_train_dev_test(authorID):
    train, test = train_test_split(authorID, test_size=0.15)
    dev, test = train_test_split(test, test_size=0.66)

    logging.info(f"{len(dev)} dev samples")
    logging.info(f"{len(test)} test samples")
    logging.info(f"{len(train)} training samples")

    return {'train': train, 'dev': dev, 'test': test}


def save_text_samples(samples_X, samples_Y, partition, input_file, output_path):
    print("saving text pair samples")
    partitions = {
        'train': ['train_candidates.jsonl', 'train_queries.jsonl'],
        'dev': ['dev_candidates.jsonl', 'dev_queries.jsonl'],
        'test': ['test_candidates.jsonl', 'test_queries.jsonl']
    }

    for key, value in partition.items():
        samples_X_partition = set(
            samples_X[samples_X['authorID'].isin(partition[key])]['documentID'].astype('str').tolist())
        samples_Y_partition = set(
            samples_Y[samples_Y['authorID'].isin(partition[key])]['documentID'].astype('str').tolist())

        with open(input_file, 'r') as f, \
                open(os.path.join(output_path, partitions[key][0]), 'w') as out1, \
                open(os.path.join(output_path, partitions[key][1]), 'w') as out2:

            for line in tqdm(f):
                line = json.loads(line)
                if line['documentID'] in samples_X_partition:
                    out1.write(json.dumps(line, ensure_ascii=False) + '\n')
                elif line['documentID'] in samples_Y_partition:
                    out2.write(json.dumps(line, ensure_ascii=False) + '\n')
