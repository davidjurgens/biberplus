import os
import sys

from align import align
from merge import merge_datasets
from partition import partition

sys.path.append('../../..')

from src.build_datasets.contrastive.convert_to_datasets import create_dataset


def build_dataset(input_file, output_path, nrows):
    partition(input_file, output_path, nrows)

    for type in ['dev', 'test', 'train']:
        build_data_split(type, output_path)


def build_data_split(type, output_path):
    print(f"Aligning {type}")
    align(os.path.join(output_path, f'{type}_candidates.jsonl'), os.path.join(output_path, f'{type}_queries.jsonl'))

    print(f"Merging {type} data")
    merge_datasets(
        [(os.path.join(output_path, f'{type}_queries.jsonl'), os.path.join(output_path, f'{type}_candidates.jsonl'))],
        os.path.join(output_path, f'{type}.jsonl'))

    print(f'Creating {type} dataset...')
    create_dataset(os.path.join(output_path, f'{type}.jsonl'), os.path.join(output_path, type))


if __name__ == '__main__':
    input_output_pairs = [
        ('/home/kalkiek/datasets/pan20-av-training-small/pan_small.jsonl',
         '/home/kalkiek/datasets/pan20-av-training-small/'),
        ('/home/kalkiek/datasets/pan20-av-training-large/pan_large.jsonl',
         '/home/kalkiek/datasets/pan20-av-training-large/'),
    ]

    nrows = None  # None to use the whole file. Samples from the first nrows

    for input_file, output_path in input_output_pairs:
        print(f"Starting on {input_file}")
        build_dataset(input_file, output_path, nrows)
