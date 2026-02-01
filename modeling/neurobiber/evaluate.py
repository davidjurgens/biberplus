import os

import sys
import numpy as np
import pandas as pd
import torch
import json

from tqdm import tqdm

os.chdir("/home/kalkiek/projects/biber-multidimensional-register-analysis/")
sys.path.append("/home/kalkiek/projects/biber-multidimensional-register-analysis/")

from modeling.neurobiber.tagger import (
    load_model_and_tokenizer,
    get_predictions,
    tag_jsonl_file,
)

# Configuration
TEST_FP = "/shared/3/projects/hiatus/tagged_data/biber-aggregate/binary_test.jsonl"
RESULTS_DIR = "/shared/3/projects/hiatus/tagged_data/biber-aggregate/evaluation/"
BATCH_SIZE = 64


def tag_test_json(
    input_file, output_dir, batch_size=32, text_key="text", show_progress=True
):
    model, tokenizer = load_model_and_tokenizer()

    # Create output files
    pred_file = os.path.join(output_dir, "predictions.npy")
    label_file = os.path.join(output_dir, "labels.npy")

    # Initialize counters for total predictions
    total_samples = 0

    with open(input_file, "r", encoding="utf-8") as fin:
        batch = []
        label_batch = []

        iterator = tqdm(fin, desc="Processing texts") if show_progress else fin

        for line in iterator:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                feature_keys = list(data["features"].keys())
                label_keys = [k for k in feature_keys if k.endswith("_mean")]

                labels = [
                    1.0 if float(data["features"][key]) != 0.0 else 0.0
                    for key in label_keys
                ]

                batch.append(data[text_key])
                label_batch.append(labels)

                if len(batch) >= batch_size:
                    predictions = get_predictions(model, batch, tokenizer)

                    # Convert to numpy arrays
                    pred_array = np.array(predictions)
                    label_array = np.array(label_batch)

                    # Append to files
                    with open(pred_file, "ab") as f:
                        np.save(f, pred_array)
                    with open(label_file, "ab") as f:
                        np.save(f, label_array)

                    total_samples += len(batch)
                    batch = []
                    label_batch = []

            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")
                continue

        # Handle remaining batch
        if batch:
            predictions = get_predictions(model, batch, tokenizer)

            pred_array = np.array(predictions)
            label_array = np.array(label_batch)

            with open(pred_file, "ab") as f:
                np.save(f, pred_array)
            with open(label_file, "ab") as f:
                np.save(f, label_array)

            total_samples += len(batch)

    print(f"Processed {total_samples} samples total")
    print(f"Results saved to {output_dir}")
    return pred_file, label_file


tag_test_json(TEST_FP, RESULTS_DIR)
