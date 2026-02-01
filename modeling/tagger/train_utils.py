import pandas as pd
from typing import Optional, Union
from transformers import Trainer
from torch.utils.data import IterableDataset
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score,
)
import json
import torch
import numpy as np


def chunk_df_iterator(file_path, chunk_size=1000):
    try:
        for chunk in pd.read_json(file_path, lines=True, chunksize=chunk_size):
            yield chunk
    except Exception as e:
        raise RuntimeError(f"Error reading file {file_path}: {str(e)}")


class StreamingDataset(IterableDataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load first row to get label columns
        with open(data_path, "r") as f:
            first_row = json.loads(f.readline())
            self.label_columns = list(first_row["features"].keys())

        # Get total rows for length calculation
        self.total_rows = sum(1 for _ in open(self.data_path))

    def __iter__(self):
        with open(self.data_path, "r") as f:
            for line in f:
                item = json.loads(line)

                # Tokenize text
                encoding = self.tokenizer(
                    item["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                # Add labels
                labels = [float(item["features"][col]) for col in self.label_columns]
                encoding["labels"] = torch.tensor(labels, dtype=torch.float)

                # Remove batch dimension added by tokenizer
                encoding = {k: v.squeeze(0) for k, v in encoding.items()}

                yield encoding

    def __len__(self):
        return self.total_rows


def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    if predictions.shape != labels.shape:
        raise ValueError(
            f"Predictions shape {predictions.shape} doesn't match labels shape {labels.shape}"
        )

    binary_predictions = (predictions != 0).astype(int)
    binary_labels = (labels != 0).astype(int)

    return {
        # Regression metrics
        "mse": mean_squared_error(labels, predictions),
        "rmse": mean_squared_error(labels, predictions, squared=False),
        "mae": mean_absolute_error(labels, predictions),
        "r2": r2_score(labels, predictions),
        # Classification metrics
        "precision_micro": precision_score(
            binary_labels, binary_predictions, average="micro"
        ),
        "recall_micro": recall_score(
            binary_labels, binary_predictions, average="micro"
        ),
        "f1_micro": f1_score(binary_labels, binary_predictions, average="micro"),
        "precision_macro": precision_score(
            binary_labels, binary_predictions, average="macro"
        ),
        "recall_macro": recall_score(
            binary_labels, binary_predictions, average="macro"
        ),
        "f1_macro": f1_score(binary_labels, binary_predictions, average="macro"),
    }


def compute_tag_level_metrics(predictions, labels, tag_names):
    tag_metrics = {}
    for i, tag in enumerate(tag_names):
        tag_predictions = predictions[:, i]
        tag_labels = labels[:, i]

        # Convert to binary for classification metrics
        binary_predictions = (tag_predictions != 0).astype(int)
        binary_labels = (tag_labels != 0).astype(int)

        tag_metrics[tag] = {
            # Regression metrics
            "mse": mean_squared_error(tag_labels, tag_predictions),
            "rmse": mean_squared_error(tag_labels, tag_predictions, squared=False),
            "mae": mean_absolute_error(tag_labels, tag_predictions),
            "r2": r2_score(tag_labels, tag_predictions),
            # Classification metrics
            "precision": precision_score(binary_labels, binary_predictions),
            "recall": recall_score(binary_labels, binary_predictions),
            "f1": f1_score(binary_labels, binary_predictions),
        }
    return tag_metrics


class EvalSamplingTrainer(Trainer):
    def __init__(self, eval_sample_ratio=0.05, random_seed=42, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_sample_ratio = eval_sample_ratio
        self.random_seed = random_seed

    def evaluate(
        self, eval_dataset: Optional[Union[StreamingDataset, None]] = None, **kwargs
    ):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        np.random.seed(self.random_seed)

        # Create a sampled version of the dataset by wrapping the iterator
        original_iter = eval_dataset.__iter__

        def sampled_iter():
            for item in original_iter():
                if np.random.random() < self.eval_sample_ratio:
                    yield item

        # Better to use a context manager pattern
        from contextlib import contextmanager

        @contextmanager
        def temporary_iterator():
            original_iter = eval_dataset.__iter__
            eval_dataset.__iter__ = sampled_iter
            try:
                yield
            finally:
                eval_dataset.__iter__ = original_iter

        with temporary_iterator():
            metrics = super().evaluate(eval_dataset, **kwargs)

        return metrics
