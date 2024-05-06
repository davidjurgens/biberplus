import jsonlines
import shutil
import logging
import os

from glob import glob
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def partition_file(input_file, output_directory, chunks=100):
    """Split the input file into smaller chunks."""
    lines = count_lines(input_file)
    logging.info(f"{lines} lines in file")

    chunk_size = round(lines / chunks)
    logging.info(f"Chunk size: {chunk_size} lines")

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

    # Save the remaining lines, if any.
    if curr_lines:
        save_partition(curr_lines, output_directory, chunk + 1)


def save_partition(json_lines, output_directory, index):
    """Save the current partition of lines to a file."""
    out = os.path.join(output_directory, f"partition-{index}.jsonl")
    logging.info(f"Saving {out}")

    with jsonlines.open(out, mode='w') as writer:
        writer.write_all(json_lines)


def count_lines(input_file):
    """Count the number of lines in the file."""
    with open(input_file, 'rb') as f:
        return sum(1 for _ in f)


def join_tagged_files(input_directory, output_file):
    """Join all tagged files from the input directory into one output file."""
    tagged_files = glob(os.path.join(input_directory, "*.jsonl"))

    print(len(tagged_files))

    with jsonlines.open(output_file, mode='w') as writer:
        for tagged_file in tqdm(tagged_files, desc="Merging tagged files"):
            with jsonlines.open(tagged_file) as reader:
                for obj in reader:
                    writer.write(obj)


def delete_partitioned_files(dir_path):
    """Delete all partitioned files in the directory."""
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        logging.error(f"Error deleting directory {dir_path}: {e.strerror}")