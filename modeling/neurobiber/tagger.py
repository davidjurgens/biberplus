import os
import sys
import json
from itertools import islice

import torch
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Configuration
MODEL_DIR = "/shared/3/projects/hiatus/tagged_data/models/roberta-base/binary-finetune-full/best_model"
CHUNK_SIZE = 512  # Defined during training
BIBER_FEATURES = [
    "BIN_QUAN",
    "BIN_QUPR",
    "BIN_AMP",
    "BIN_PASS",
    "BIN_XX0",
    "BIN_JJ",
    "BIN_BEMA",
    "BIN_CAUS",
    "BIN_CONC",
    "BIN_COND",
    "BIN_CONJ",
    "BIN_CONT",
    "BIN_DPAR",
    "BIN_DWNT",
    "BIN_EX",
    "BIN_FPP1",
    "BIN_GER",
    "BIN_RB",
    "BIN_PIN",
    "BIN_INPR",
    "BIN_TO",
    "BIN_NEMD",
    "BIN_OSUB",
    "BIN_PASTP",
    "BIN_VBD",
    "BIN_PHC",
    "BIN_PIRE",
    "BIN_PLACE",
    "BIN_POMD",
    "BIN_PRMD",
    "BIN_WZPRES",
    "BIN_VPRT",
    "BIN_PRIV",
    "BIN_PIT",
    "BIN_PUBV",
    "BIN_SPP2",
    "BIN_SMP",
    "BIN_SERE",
    "BIN_STPR",
    "BIN_SUAV",
    "BIN_SYNE",
    "BIN_TPP3",
    "BIN_TIME",
    "BIN_NOMZ",
    "BIN_BYPA",
    "BIN_PRED",
    "BIN_TOBJ",
    "BIN_TSUB",
    "BIN_THVC",
    "BIN_NN",
    "BIN_DEMP",
    "BIN_DEMO",
    "BIN_WHQU",
    "BIN_EMPH",
    "BIN_HDG",
    "BIN_WZPAST",
    "BIN_THAC",
    "BIN_PEAS",
    "BIN_ANDC",
    "BIN_PRESP",
    "BIN_PROD",
    "BIN_SPAU",
    "BIN_SPIN",
    "BIN_THATD",
    "BIN_WHOBJ",
    "BIN_WHSUB",
    "BIN_WHCL",
    "BIN_ART",
    "BIN_AUXB",
    "BIN_CAP",
    "BIN_SCONJ",
    "BIN_CCONJ",
    "BIN_DET",
    "BIN_EMOJ",
    "BIN_EMOT",
    "BIN_EXCL",
    "BIN_HASH",
    "BIN_INF",
    "BIN_UH",
    "BIN_NUM",
    "BIN_LAUGH",
    "BIN_PRP",
    "BIN_PREP",
    "BIN_NNP",
    "BIN_QUES",
    "BIN_QUOT",
    "BIN_AT",
    "BIN_SBJP",
    "BIN_URL",
    "BIN_WH",
    "BIN_INDA",
    "BIN_ACCU",
    "BIN_PGAS",
    "BIN_CMADJ",
    "BIN_SPADJ",
    "BIN_X",
]


def predict_text(model, tokenizer, text, chunk_size=CHUNK_SIZE, subbatch_size=32):
    return get_predictions_chunked_batch(
        model, tokenizer, [text], chunk_size, subbatch_size
    )[0]


def predict_batch(model, tokenizer, texts, chunk_size=CHUNK_SIZE, subbatch_size=32):
    return get_predictions_chunked_batch(
        model, tokenizer, texts, chunk_size, subbatch_size
    )


def chunk_text(text, chunk_size=CHUNK_SIZE):
    tokens = text.strip().split()
    if not tokens:
        return []
    return [
        " ".join(tokens[i : i + chunk_size]) for i in range(0, len(tokens), chunk_size)
    ]


def get_predictions_chunked_batch(
    model, tokenizer, texts, chunk_size=CHUNK_SIZE, subbatch_size=32
):
    # Map each text to its chunks while preserving order
    chunked_texts = []
    chunk_indices = []  # [(start, end)] indices for each original text's chunks
    for idx, text in enumerate(texts):
        start = len(chunked_texts)
        text_chunks = chunk_text(text, chunk_size)
        chunked_texts.extend(text_chunks)
        chunk_indices.append(
            {"original_idx": idx, "chunk_range": (start, start + len(text_chunks))}
        )

    # Handle empty batch case
    if not chunked_texts:
        return np.zeros((len(texts), model.config.num_labels))

    # Process chunks in subbatches
    all_chunk_preds = []
    for i in range(0, len(chunked_texts), subbatch_size):
        batch_chunks = chunked_texts[i : i + subbatch_size]
        encodings = tokenizer(
            batch_chunks,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=chunk_size,
        ).to("cuda")

        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs = model(**encodings)
            probs = torch.sigmoid(outputs.logits)
        all_chunk_preds.append(probs.cpu())

    # Combine all predictions
    all_chunk_preds = (
        torch.cat(all_chunk_preds, dim=0) if all_chunk_preds else torch.empty(0)
    )

    # Aggregate predictions for each text in original order
    predictions = [None] * len(texts)  # Pre-allocate list with correct size
    for info in chunk_indices:
        start, end = info["chunk_range"]
        if start == end:  # Empty text case
            pred = torch.zeros(model.config.num_labels)
        else:
            # Take max probability across chunks for each feature
            chunk_preds = all_chunk_preds[start:end]
            pred, _ = torch.max(chunk_preds, dim=0)
        predictions[info["original_idx"]] = (pred > 0.5).int().numpy()

    return np.array(predictions)


def tag_jsonl_file(
    input_file, output_file, batch_size=32, text_key="fullText", show_progress=True
):
    model, tokenizer = load_model_and_tokenizer()

    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        batch = []
        iterator = tqdm(fin, desc="Processing texts") if show_progress else fin
        for line in iterator:
            try:
                item = json.loads(line)
                batch.append(item)
                if len(batch) >= batch_size:
                    try:
                        texts = [item[text_key] for item in batch]
                        predictions = predict_batch(model, tokenizer, texts)
                        for item, pred in zip(batch, predictions):
                            output_item = {
                                "documentID": item["documentID"],
                                "neural_biber": pred.tolist(),
                            }
                            fout.write(json.dumps(output_item) + "\n")
                    except Exception as e:
                        print(f"Error processing batch: {str(e)}")
                    batch = []
            except json.JSONDecodeError as e:
                print(f"Error reading JSON line: {str(e)}")
                continue

        if batch:
            try:
                texts = [item[text_key] for item in batch]
                predictions = predict_batch(model, tokenizer, texts)
                for item, pred in zip(batch, predictions):
                    output_item = {
                        "documentID": item["documentID"],
                        "neural_biber": pred.tolist(),
                    }
                    fout.write(json.dumps(output_item) + "\n")
            except Exception as e:
                print(f"Error processing final batch: {str(e)}")


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to("cuda")
    model.eval()
    return model, tokenizer
