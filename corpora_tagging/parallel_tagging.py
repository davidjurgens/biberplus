import os
import jsonlines

from glob import glob
from multiprocessing import Pool
from biberplus.reducer import encode_text
from biberplus.tagger.tagger_utils import load_tokenizer


def dict_add(dictionary: dict, outer_key, inner_key):
    if outer_key not in dictionary:
        dictionary[outer_key] = {}
    if inner_key not in dictionary[outer_key]:
        dictionary[outer_key][inner_key] = 0
    dictionary[outer_key][inner_key] += 1


def tag_partitions(config, input_directory, output_directory, num_workers, post_counts, default_niceness=20):
    process_args = build_process_args(config, input_directory, output_directory)

    def set_niceness():
        os.nice(default_niceness)

    with Pool(num_workers, initializer=set_niceness) as pool:
        pool.starmap(tag_partition, process_args)


def tag_partition(config, input_file, output_file, post_counts):
    print(f"Tagging file {input_file}\n")
    tokenizer = load_tokenizer(use_gpu=False)
    tagged_objects = []

    with jsonlines.open(input_file) as reader:
        for obj in reader:
            tagged_object = tag_object(obj, config, tokenizer)
            tagged_objects.append(tagged_object)


            dict_add(post_counts, obj["author"], obj["subreddit"])
            # author_subreddit = (obj['author'], obj['subreddit'])
            # if author_subreddit in post_counts:
            #     post_counts[author_subreddit] += 1
            # else:
            #     post_counts[author_subreddit] = 1

            if len(tagged_objects) % 5000 == 0:
                append_chunk(output_file, tagged_objects)
                tagged_objects = []

    if tagged_objects:
        append_chunk(output_file, tagged_objects)


def write_counts_to_tsv(post_counts, counts_file_name):
    with open(counts_file_name, 'w') as f:
        f.write("author\tsubreddit\tpost_count\n")  # Header
        for (author, subreddit), count in post_counts.items():
            f.write(f"{author}\t{subreddit}\t{count}\n")


def build_process_args(config, input_directory, output_directory):
    partition_files = glob(f"{input_directory}*.jsonl")
    process_args = []

    for fp in partition_files:
        fname = fp.rsplit('/', 1)[-1].replace('.jsonl', '') + '-tagged.jsonl'
        out = f"{output_directory}{fname}"
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
    with jsonlines.open(output_file, mode='a') as writer:
        writer.write_all(tagged_objects)
