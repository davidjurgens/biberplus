"""Authorship comparison: same-author probability for two texts from a
random forest over [A | B | |A-B|] of their 96-dim presence vectors,
trained on PAN 2020 (small).

The forest is not shipped in the wheel (~50 MB); it downloads once from the
Hugging Face Hub and is cached. Override with the NEUROBIBER_AV_PATH env
var (a local .pkl) or NEUROBIBER_AV_REPO (an alternative Hub repo).

Not forensic-grade: accuracy in the ~0.6-0.8 weighted-F1 range on PAN 2020
is useful signal, never sole evidence of authorship. The loaded model's own
evaluation metrics are in `metrics()`.
"""
import os
import pickle
import threading

import numpy as np

from biberplus.neurobiber.features import feature_codes, name_of

AV_REPO = os.environ.get("NEUROBIBER_AV_REPO", "Blablablab/neurobiber-authorship")
AV_FILENAME = "rf_av.pkl"

_lock = threading.Lock()
_state = {}


def _artifact_path():
    local = os.environ.get("NEUROBIBER_AV_PATH")
    if local:
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(AV_REPO, AV_FILENAME)


def _load():
    with _lock:
        if "art" not in _state:
            with open(_artifact_path(), "rb") as f:
                art = pickle.load(f)
            assert art["feature_codes"] == feature_codes(), \
                "authorship model feature order does not match this package"
            _state["art"] = art
    return _state["art"]


def pair_features(vec_a, vec_b):
    """[A | B | |A-B|] over two 96-dim vectors -> (288,)."""
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    return np.concatenate([a, b, np.abs(a - b)])


def same_author_probability(vec_a, vec_b):
    art = _load()
    x = pair_features(vec_a, vec_b).reshape(1, -1)
    return float(art["clf"].predict_proba(x)[0, 1])


def metrics():
    """The loaded model's own evaluation record (weighted F1, ROC-AUC,
    train/test sizes, provenance note)."""
    return _load().get("metrics", {})


def drivers(vec_a, vec_b, top_n=8):
    """Features most responsible for a verdict: random-forest importance
    (summed over each feature's A/B/|A-B| copies) plus this pair's actual
    agreement. Agreement on a high-importance feature is same-author
    evidence; disagreement is different-author evidence.

    Returns [{code, name, a, b, agree, importance}] by importance."""
    art = _load()
    imp = np.asarray(art["clf"].feature_importances_)
    n = len(art["feature_codes"])
    per_feature = imp[:n] + imp[n:2 * n] + imp[2 * n:3 * n]

    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    order = np.argsort(per_feature)[::-1]
    out = []
    for i in order[:top_n]:
        code = art["feature_codes"][i]
        out.append({"code": code, "name": name_of(code),
                    "a": int(round(a[i])), "b": int(round(b[i])),
                    "agree": bool(round(a[i]) == round(b[i])),
                    "importance": float(per_feature[i])})
    return out


def compare_texts(text_a, text_b, **predict_kwargs):
    """End to end: tag both texts and return verdict + drivers as a dict."""
    from biberplus.neurobiber.model import predict_binary
    vecs = predict_binary([text_a, text_b], **predict_kwargs).astype(float)
    prob = same_author_probability(vecs[0], vecs[1])
    return {
        "same_author_probability": round(prob, 3),
        "verdict": "same author" if prob >= 0.5 else "different authors",
        "threshold": 0.5,
        "drivers": drivers(vecs[0], vecs[1]),
        "model_metrics": metrics(),
        "caveat": "Not forensic-grade; never sole evidence of authorship.",
    }
