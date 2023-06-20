import jsonlines
import shutil

from glob import glob
from tqdm import tqdm


def partition_file(input_file, output_directory, chunks=100):
    lines = count_lines(input_file)
    print(f"{lines} lines in file")
    chunk_size = round(lines / chunks)
    print(f"Chunk size: {chunk_size} lines")

    curr_lines = []
    count, chunk = 0, 1

    with jsonlines.open(input_file) as reader:
        for obj in reader:
            curr_lines.append(obj)
            count += 1
            if count >= chunk_size:
                save_partition(curr_lines, output_directory, chunk)
                count = 0
                curr_lines = []
                chunk += 1

    if curr_lines:
        save_partition(curr_lines, output_directory, chunk + 1)


def save_partition(json_lines, output_directory, index):
    out = f"{output_directory}partition-{index}.jsonl"
    print(f"Saving {out}")

    with jsonlines.open(out, mode='w') as writer:
        writer.write_all(json_lines)


def count_lines(input_file):
    return sum(1 for _ in open(input_file, 'rb'))


def join_tagged_files(input_directory, output_file):
    tagged_files = glob(input_directory + "*-tagged.jsonl")
    with jsonlines.open(output_file, mode='w') as writer:
        for tagged_file in tqdm(tagged_files):
            with jsonlines.open(tagged_file) as reader:
                for obj in reader:
                    writer.write(obj)


def delete_partitioned_files(dir_path):
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))