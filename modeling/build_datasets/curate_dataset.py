import json
import os

import jsonlines
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def count_lines(input_file):
    """Returns the number of lines in a file"""
    print(f"Counting lines in {input_file}")
    try:
        with open(input_file, "rb") as file:
            return sum(1 for _ in file)
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        raise


def get_encodings_df(input_file, total_lines):
    """Returns a DataFrame from provided rows"""
    print(f"Building encodings for {input_file}")
    rows = []

    with jsonlines.open(input_file) as reader:
        for index, obj in tqdm(enumerate(reader), total=total_lines):
            if "encodings" in obj and obj["encodings"]:
                row = [index, obj["documentID"]]
                row.extend(obj["encodings"]["binary"])
                rows.append(row)

    columns = ["index", "documentID"] + list(range(192))
    return pd.DataFrame(rows, columns=columns)


def stratified_sample(df, sample_size):
    """Perform stratified sampling on the dataframe."""
    documentIDs = set()

    for column in tqdm(
        df.columns.tolist(), desc="Stratified sampling. No feature left behind"
    ):
        if column in ["index", "documentID"]:
            continue
        col_df = df[df[column] != 0]
        if len(col_df) < sample_size:
            sample_documents = set(col_df["documentID"].unique())
        else:
            sample_df = col_df.sample(sample_size, replace=False, random_state=0)
            sample_documents = set(sample_df["documentID"].unique())
        documentIDs.update(sample_documents)

    return documentIDs


def save_samples(input_file, output_file, total_lines, sample_documentIDs):
    """Save samples from the input file to the output file."""
    num_docs = len(sample_documentIDs)
    print(f"Appending {num_docs} document samples from {input_file} to {output_file}")
    try:
        with jsonlines.open(output_file, mode="a") as writer:
            with jsonlines.open(input_file) as reader:
                for obj in tqdm(
                    reader,
                    desc=f"Saving {num_docs} to {output_file}",
                    total=total_lines,
                ):
                    if obj["documentID"] in sample_documentIDs:
                        line = {
                            "text": obj["fullText"],
                            "documentID": obj["documentID"],
                            "text_biberPlus": obj["encodings"]["binary"],
                            "authorID": obj["authorIDs"],
                        }
                        writer.write(line)
    except Exception as e:
        print("An error occurred while saving samples:", str(e))


def curate(input_file, intermediate_encodings, output_file, K=10000):
    total_lines = count_lines(input_file)
    print(f"{total_lines} lines in {input_file}")

    if os.path.isfile(intermediate_encodings):
        print(f"Loading encodings from {intermediate_encodings}")
        encodings_df = pd.read_parquet(intermediate_encodings)
    else:
        encodings_df = get_encodings_df(input_file, total_lines)
        print(f"Saving binary encodings to {intermediate_encodings}")
        encodings_df.to_parquet(intermediate_encodings, index=False, compression="gzip")

    sample_documentIDs = stratified_sample(encodings_df, K)
    save_samples(input_file, output_file, total_lines, sample_documentIDs)


def split_dataset(input_file, train_file, dev_file, test_file, test_size=0.15):
    total_lines = count_lines(input_file)
    print(f"Total lines {total_lines}")

    # Calculate sizes of splits
    test_lines = int(total_lines * test_size)
    dev_lines = int(test_lines * 0.66)
    train_lines = total_lines - test_lines - dev_lines
    print(f"Train lines: {train_lines} Dev lines: {dev_lines} Test lines: {test_lines}")

    # Generate shuffled indices
    indices = np.random.permutation(total_lines)
    train_indices, dev_indices, test_indices = np.split(
        indices, [train_lines, train_lines + dev_lines]
    )

    # Convert to set for faster lookup
    train_indices = set(train_indices)
    dev_indices = set(dev_indices)

    with (
        jsonlines.open(input_file, "r") as reader,
        open(train_file, "w") as train_writer,
        open(dev_file, "w") as dev_writer,
        open(test_file, "w") as test_writer,
    ):
        for idx, obj in tqdm(enumerate(reader), total=total_lines):
            line = json.dumps(
                {
                    "text": obj["text"],
                    "documentID": str(obj["documentID"]),
                    "authorID": str(obj["authorID"]),
                    "text_biberPlus": obj["text_biberPlus"],
                },
                ensure_ascii=False,
            )
            if idx in train_indices:
                train_writer.write(line + "\n")
            elif idx in dev_indices:
                dev_writer.write(line + "\n")
            else:
                test_writer.write(line + "\n")


def create_datasets(datasets, folder):
    for dataset in datasets:
        print(f"Creating {dataset} dataset...")
        inpath = os.path.join(folder, f"{dataset}.jsonl")
        outpath = os.path.join(folder, dataset)
        create_dataset(inpath, outpath)


def create_dataset(inpath, outpath):
    dataset = load_dataset("json", data_files=inpath)
    dataset.save_to_disk(outpath)


if __name__ == "__main__":
    input_files = [
        "/shared/3/projects/hiatus/tagged_data/amazon/amazon.jsonl",
        "/shared/3/projects/hiatus/tagged_data/reddit/reddit.jsonl",
        "/shared/3/projects/hiatus/tagged_data/book3corpus/book3corpus.jsonl",
        "/shared/3/projects/hiatus/tagged_data/wiki/wiki.jsonl",
        "/shared/3/projects/hiatus/tagged_data/wiki_discussions/wiki_discussions.jsonl",
        "/shared/3/projects/hiatus/tagged_data/realnews/realnews.jsonl",
        "/shared/3/projects/hiatus/tagged_data/gmane/gmane.jsonl",
    ]

    intermediate_encoding_files = [
        "/shared/3/projects/hiatus/tagged_data/amazon/binary_encodings.parquet.gzip",
        "/shared/3/projects/hiatus/tagged_data/reddit/binary_encodings.parquet.gzip",
        "/shared/3/projects/hiatus/tagged_data/book3corpus/binary_encodings.parquet.gzip",
        "/shared/3/projects/hiatus/tagged_data/wiki/binary_encodings.parquet.gzip",
        "/shared/3/projects/hiatus/tagged_data/wiki_discussions/binary_encodings.parquet.gzip",
        "/shared/3/projects/hiatus/tagged_data/realnews/binary_encodings.parquet.gzip",
        "/shared/3/projects/hiatus/tagged_data/gmane/binary_encodings.parquet.gzip",
    ]

    output_file = "/shared/3/projects/hiatus/tagged_data/mlm_finetuning/corpus.jsonl"

    # Save a stratified sample of Biber features from each dataset
    for input_file, intermediate_encodings in list(
        zip(input_files, intermediate_encoding_files)
    ):
        curate(input_file, intermediate_encodings, output_file)

    # Shuffle and split into multiple files
    input_file = "/shared/3/projects/hiatus/tagged_data/mlm_finetuning/corpus.jsonl"
    train_file = "/shared/3/projects/hiatus/tagged_data/mlm_finetuning/train.jsonl"
    dev_file = "/shared/3/projects/hiatus/tagged_data/mlm_finetuning/dev.jsonl"
    test_file = "/shared/3/projects/hiatus/tagged_data/mlm_finetuning/test.jsonl"

    split_dataset(input_file, train_file, dev_file, test_file)

    folder = "/shared/3/projects/hiatus/tagged_data/mlm_finetuning/"
    datasets = ["train", "dev", "test"]

    create_datasets(datasets, folder)
