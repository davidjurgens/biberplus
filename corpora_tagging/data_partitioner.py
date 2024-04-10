import shutil
import logging
import os

from glob import glob
from tqdm import tqdm
import gzip

logging.basicConfig(level=logging.INFO)


def partition_file(input_file, output_directory, chunks=100):
    """Split the input file into smaller chunks."""
    lines = count_lines(input_file)
    logging.info(f"{lines} lines in file")

    chunk_size = round(lines / chunks)
    logging.info(f"Chunk size: {chunk_size} lines")

    curr_lines = []
    count, chunk = 0, 1

    with gzip.open(input_file) as f:
        for obj in f:
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
    os.makedirs(output_directory, exist_ok=True)
    
    out_file = os.path.join(output_directory, f"partition-{index}.jsonl.gz")
    logging.info(f"Saving {out_file}")

    with gzip.open(out_file, mode='wb') as f:
        for line in json_lines:
            f.write(line)


def count_lines(input_file):
    """Count the number of lines in the file."""
    with gzip.open(input_file, 'rb') as f:
        return sum(1 for _ in f)



def join_tagged_files(input_directory, output_file):
    """Join all tagged .gzip files from the input directory into one output .gzip file."""
    # Adjust the pattern to match .gzip files
    tagged_files = glob(os.path.join(input_directory, "*.gz"))

    # Open the output file with gzip in write mode
    with gzip.open(output_file, mode='wt', encoding='utf-8') as writer:
        for tagged_file in tqdm(tagged_files, desc="Merging tagged files"):
            # Open each tagged file with gzip in read mode
            with gzip.open(tagged_file, mode='rt', encoding='utf-8') as reader:
                for line in reader:
                    # Each line is a JSON string, so we can write it directly
                    writer.write(line)

def delete_partitioned_files(dir_path):
    """Delete all partitioned files in the directory."""
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        logging.error(f"Error deleting directory {dir_path}: {e.strerror}")
