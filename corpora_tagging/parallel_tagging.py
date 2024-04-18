import os

from glob import glob
from multiprocessing import Pool
from biberplus.reducer import encode_text
from biberplus.tagger.tagger_utils import load_tokenizer
import gzip
import json
from tqdm import tqdm

def dict_add(dictionary: dict, outer_key, inner_key, count = 1):
    if outer_key not in dictionary:
        dictionary[outer_key] = {}
    if inner_key not in dictionary[outer_key]:
        dictionary[outer_key][inner_key] = 0
    dictionary[outer_key][inner_key] += count



def tag_partitions(config, input_directory, output_directory, num_workers, post_counts, default_niceness=20):
    process_args = build_process_args(config, input_directory, output_directory)
    # print(process_args)
    def set_niceness():
        os.nice(default_niceness)

    with Pool(num_workers, initializer=set_niceness) as pool:
        results = pool.starmap(tag_partition, process_args)

    for local_count in results:
          for author, subreddit_dict in local_count.items():
              for subreddit, count in subreddit_dict.items():
                dict_add(post_counts, author, subreddit, count)

def tag_partition(config, input_file, output_file):
    post_counts = {}
    print(f"Tagging file {input_file}\n")
    tokenizer = load_tokenizer(use_gpu=False)
    tagged_objects = []

    with gzip.open(input_file, 'rb') as f:
        # Decompress and decode the file
        decompressed_data = f.read().decode('utf-8')
        
        # Split the data into individual samples based on newline delimiter
        samples = decompressed_data.split('\n')
        
        # Iterate over individual samples with a progress bar
        for obj_str in samples:
            if not obj_str.strip():  # Skip empty lines
                continue
            obj = json.loads(obj_str)
            dict_add(post_counts, obj["author"], obj["subreddit"])
            tagged_object = tag_object(obj, config, tokenizer)
            tagged_objects.append(tagged_object)

            if len(tagged_objects) % 1000 == 0:
                append_chunk(output_file, tagged_objects)
                tagged_objects = []

    # Handling the last batch if it didn't reach the batch size threshold
    if tagged_objects:
        append_chunk(output_file, tagged_objects)
    
    return post_counts


def write_to_tsv(post_counts, counts_file_name):
    with open(counts_file_name, 'w') as f:
        f.write("author\tsubreddit\tpost_count\n")  # Header
        for author, subreddit_dict in post_counts.items():
            for subreddit, count in subreddit_dict.items():
                f.write(f"{author}\t{subreddit}\t{count}\n")


def build_process_args(config, input_directory, output_directory):
    partition_files = glob(os.path.join(input_directory, '*.gz'))
    process_args = []

    for fp in partition_files:
        fname = fp.rsplit('/')[-1].replace('.gz', '') + '-tagged.gz'
        out = os.path.join(output_directory, fname)
        process_args.append((config, fp, out))

    return process_args


def tag_object(obj, config, tokenizer):
    try:
        num_tokens = len(tokenizer(obj['body']))
        obj['num_tokens'] = num_tokens
        if num_tokens >= 10:
            obj['encodings'] = encode_text(text=obj['body'], config=config)
    except Exception:
        pass

    return obj


def append_chunk(output_file, tagged_objects):
    # Opening the file in append mode
    with gzip.open(output_file, mode='ab') as writer:
        for obj in tagged_objects:
            writer.write((json.dumps(obj) + "\n").encode('utf-8'))  # Writing each object as a new line
