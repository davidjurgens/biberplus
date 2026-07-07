"""Neurobiber inference: 96-dim presence vectors (and probabilities).

Reference implementation semantics: texts are split into consecutive
512-whitespace-token chunks, each chunk scored with sigmoid(logits), and
chunk probabilities combined with an element-wise max (equivalent to OR
after thresholding at 0.5). Requires `pip install biberplus[neural]`.
"""
import threading

import numpy as np
import torch

MODEL_NAME = "Blablablab/neurobiber"
CHUNK_SIZE = 512  # whitespace tokens, matches training
THRESHOLD = 0.5

_lock = threading.Lock()
_state = {}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name=MODEL_NAME, device=None):
    """Load model + tokenizer once per process (thread-safe)."""
    with _lock:
        if _state.get("name") != model_name:
            from transformers import (AutoModelForSequenceClassification,
                                      AutoTokenizer)
            device = device or get_device()
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.to(device)
            model.eval()
            _state.update(name=model_name, model=model, tokenizer=tokenizer,
                          device=device)
    return _state["model"], _state["tokenizer"], _state["device"]


def chunk_text(text, chunk_size=CHUNK_SIZE):
    tokens = text.strip().split()
    if not tokens:
        return []
    return [" ".join(tokens[i:i + chunk_size])
            for i in range(0, len(tokens), chunk_size)]


def predict_probs(texts, subbatch_size=16, max_chunks_per_text=None):
    """(n_texts, 96) per-feature probabilities.

    Per-text probability = max over that text's chunks. Empty texts get
    zeros. `max_chunks_per_text` truncates very long inputs.
    """
    model, tokenizer, device = load_model()

    chunked, ranges = [], []
    for text in texts:
        chunks = chunk_text(text)
        if max_chunks_per_text is not None:
            chunks = chunks[:max_chunks_per_text]
        ranges.append((len(chunked), len(chunked) + len(chunks)))
        chunked.extend(chunks)

    n_labels = model.config.num_labels
    if not chunked:
        return np.zeros((len(texts), n_labels), dtype=np.float32)

    preds = []
    for i in range(0, len(chunked), subbatch_size):
        enc = tokenizer(chunked[i:i + subbatch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=CHUNK_SIZE)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            probs = torch.sigmoid(model(**enc).logits)
        preds.append(probs.float().cpu())
    all_probs = torch.cat(preds, dim=0)

    out = np.zeros((len(texts), n_labels), dtype=np.float32)
    for row, (start, end) in enumerate(ranges):
        if start != end:
            out[row] = all_probs[start:end].max(dim=0).values.numpy()
    return out


def predict_binary(texts, **kwargs):
    """(n_texts, 96) binary presence vectors, thresholded at 0.5."""
    return (predict_probs(texts, **kwargs) > THRESHOLD).astype(np.int8)


def predict_text(text, **kwargs):
    """Convenience for one text: (binary_vector, prob_vector)."""
    probs = predict_probs([text], **kwargs)[0]
    return (probs > THRESHOLD).astype(np.int8), probs


def named_vector(text, **kwargs):
    """{feature_code: probability} for one text — the JSON-friendly form."""
    from biberplus.neurobiber.features import feature_codes
    probs = predict_probs([text], **kwargs)[0]
    return {c: round(float(p), 4) for c, p in zip(feature_codes(), probs)}
