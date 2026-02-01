import json
import subprocess
from typing import Optional, Union

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import IterableDataset
from transformers import Trainer


class StreamingDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, max_length=512, batch_size=None):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

        try:
            result = subprocess.run(
                ["wc", "-l", file_path], capture_output=True, text=True
            )
            self.total_rows = int(result.stdout.split()[0])
            print(f"Total rows in dataset: {self.total_rows}")
        except Exception:
            self.total_rows = sum(1 for _ in open(file_path))

        # Read first line to get column names
        with open(file_path, "r") as f:
            first_line = json.loads(f.readline())
            feature_columns = list(first_line["features"].keys())
            self.label_columns = [
                col[:-5] for col in feature_columns if col.endswith("_mean")
            ]

    def __iter__(self):
        with open(self.file_path, "r") as f:
            for line in f:
                data = json.loads(line)

                # Tokenize the text
                encoding = self.tokenizer(
                    data["text"],
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                # Extract mean values and convert to binary
                labels = []
                for col in self.label_columns:
                    mean_value = float(data["features"][f"{col}_mean"])
                    binary_value = 1.0 if mean_value > 0 else 0.0
                    labels.append(binary_value)

                yield {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": torch.tensor(labels, dtype=torch.float),
                }

    def __len__(self):
        return self.total_rows


def compute_metrics(eval_pred):
    predictions = torch.sigmoid(torch.tensor(eval_pred.predictions)).numpy()
    labels = eval_pred.label_ids

    # Convert logits to binary predictions
    predictions = (predictions > 0.5).astype(int)
    labels = labels.astype(int)

    return {
        "precision_micro": precision_score(
            labels, predictions, average="micro", zero_division=0
        ),
        "recall_micro": recall_score(
            labels, predictions, average="micro", zero_division=0
        ),
        "f1_micro": f1_score(labels, predictions, average="micro", zero_division=0),
        "precision_macro": precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def compute_tag_level_metrics(predictions, labels, label_columns):
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
    predictions = (predictions > 0.5).astype(np.int32)

    metrics = {}
    for i, tag in enumerate(label_columns):
        tag_preds = predictions[:, i]
        tag_labels = labels[:, i]

        metrics[tag] = {
            "precision": precision_score(tag_labels, tag_preds, zero_division=0),
            "recall": recall_score(tag_labels, tag_preds, zero_division=0),
            "f1": f1_score(tag_labels, tag_preds, zero_division=0),
        }

    return metrics


class EvalSamplingTrainer(Trainer):
    def __init__(self, eval_sample_ratio=0.01, random_seed=42, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_sample_ratio = eval_sample_ratio
        self.random_seed = random_seed
        self._sampling_stats = {"total_seen": 0, "total_sampled": 0}

    def evaluate(
        self, eval_dataset: Optional[Union[StreamingDataset, None]] = None, **kwargs
    ):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        np.random.seed(self.random_seed)

        # Reset sampling stats (handle multiple eval workers)
        self._sampling_stats = {"total_seen": 0, "total_sampled": 0}

        # Create a new dataset with sampling
        class SampledDataset(IterableDataset):
            def __init__(self, original_dataset, sample_ratio, stats_dict):
                self.original_dataset = original_dataset
                self.sample_ratio = sample_ratio
                self.length = int(len(original_dataset) * sample_ratio)
                self.stats_dict = stats_dict

            def __iter__(self):
                iterator = iter(self.original_dataset)
                for item in iterator:
                    self.stats_dict["total_seen"] += 1
                    if np.random.random() < self.sample_ratio:
                        self.stats_dict["total_sampled"] += 1
                        yield item

            def __len__(self):
                return self.length

        sampled_dataset = SampledDataset(
            eval_dataset, self.eval_sample_ratio, self._sampling_stats
        )
        metrics = super().evaluate(sampled_dataset, **kwargs)

        return metrics
