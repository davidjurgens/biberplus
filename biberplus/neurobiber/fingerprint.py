"""Style fingerprints over the CORE register space.

Everything here runs from the JSON artifact shipped with the package
(assets/core_fingerprint.json: a StandardScaler + full-rank PCA fitted on
Neurobiber presence vectors of the public CORE dev split, plus per-register
mean profiles). Only numpy is required — no torch, no sklearn.

Vectors passed in are 96-dim presence probabilities or binaries in the
canonical feature order (see biberplus.neurobiber.features). For register
placement, tag with `max_chunks_per_text=1` so your text's presence window
matches the artifact's (a document's first 512 tokens).
"""
import json
from functools import lru_cache
from importlib import resources

import numpy as np

from biberplus.neurobiber.features import feature_codes, feature_list

# Biber (1988) dimensions, scored as mean standardized presence of the
# salient positive-pole features minus the negative pole. D1's poles come
# from feature metadata; the rest are defined here.
DIMENSIONS = [
    {"key": "d1", "name": "Involved vs. informational",
     "left": "Involved", "right": "Informational", "plus": None, "minus": None},
    {"key": "d2", "name": "Narrative vs. non-narrative",
     "left": "Narrative", "right": "Non-narrative",
     "plus": {"VBD", "TPP3", "PEAS", "PUBV", "PRESP", "WZPAST"},
     "minus": {"VPRT", "JJ"}},
    {"key": "d3", "name": "Explicit vs. situation-dependent reference",
     "left": "Explicit reference", "right": "Situation-dependent",
     "plus": {"WHSUB", "WHOBJ", "PIRE", "NOMZ", "PHC"},
     "minus": {"TIME", "PLACE", "RB"}},
    {"key": "d4", "name": "Overt expression of persuasion",
     "left": "Overt persuasion", "right": "Unmarked",
     "plus": {"INF", "PRMD", "SUAV", "COND", "NEMD", "POMD", "SPAU"},
     "minus": set()},
    {"key": "d5", "name": "Abstract vs. non-abstract information",
     "left": "Abstract", "right": "Concrete",
     "plus": {"CONJ", "PASS", "BYPA", "PASTP", "WZPAST", "OSUB"},
     "minus": set()},
]


@lru_cache(maxsize=1)
def _art():
    ref = (resources.files("biberplus.neurobiber") / "assets"
           / "core_fingerprint.json")
    art = json.loads(ref.read_text())
    assert art["feature_codes"] == feature_codes(), \
        "core_fingerprint.json feature order does not match feature_meta.json"
    # ndarrays once, cached
    art["_mean"] = np.array(art["scaler_mean"])
    art["_scale"] = np.array(art["scaler_scale"])
    art["_components"] = np.array(art["components"])
    for reg in art["registers"].values():
        reg["_centroid"] = np.array(reg["centroid"])
        reg["_feature_mean"] = np.array(reg["feature_mean"])
    return art


def source():
    """Provenance string for the fitted artifact."""
    return _art()["source"]


def project(vectors):
    """(n, 96) presence vectors -> (n, 96) PCA scores."""
    art = _art()
    X = (np.asarray(vectors, dtype=float) - art["_mean"]) / art["_scale"]
    return X @ art["_components"].T


def explained_variance_ratio():
    return np.array(_art()["explained_variance_ratio"])


def loadings(pc_index, top_n=8):
    """Top positive and negative feature loadings for one PC.
    Returns (positive, negative): lists of (code, loading)."""
    comp = _art()["_components"][pc_index]
    order = np.argsort(comp)
    codes = feature_codes()
    pos = [(codes[i], float(comp[i])) for i in order[::-1][:top_n]]
    neg = [(codes[i], float(comp[i])) for i in order[:top_n]]
    return pos, neg


def nearest_registers(vector, k=3):
    """CORE registers most similar to `vector` by cosine over register mean
    profiles (cosine discounts overall presence rate, i.e. text length).
    Returns [(code, name, similarity)] best-first."""
    v = np.asarray(vector, dtype=float)
    sims = []
    for code, reg in _art()["registers"].items():
        m = reg["_feature_mean"]
        denom = np.linalg.norm(v) * np.linalg.norm(m) + 1e-9
        sims.append((float(v @ m / denom), code, reg["name"]))
    sims.sort(reverse=True)
    return [(code, name, s) for s, code, name in sims[:k]]


def register_map():
    """[(code, name, pc_scores, n_docs)] for plotting the register cloud."""
    return [(code, reg["name"], reg["_centroid"], reg["count"])
            for code, reg in _art()["registers"].items()]


def _pole_indices():
    idx = {c: i for i, c in enumerate(feature_codes())}
    d1_plus = {f["code"] for f in feature_list() if f["pole"] == "involved"}
    d1_minus = {f["code"] for f in feature_list()
                if f["pole"] == "informational"}
    out = []
    for dim in DIMENSIONS:
        plus = dim["plus"] if dim["plus"] is not None else d1_plus
        minus = dim["minus"] if dim["minus"] is not None else d1_minus
        out.append((dim, [idx[c] for c in plus if c in idx],
                    [idx[c] for c in minus if c in idx]))
    return out


def _dim_score(z, plus_idx, minus_idx):
    # Center on the vector's own mean z first: shorter inputs fire fewer
    # features across the board, which would otherwise drag every dimension
    # toward one pole. Centering keeps only the relative profile.
    z = z - z.mean()
    s = float(np.mean(z[plus_idx])) if plus_idx else 0.0
    if minus_idx:
        s -= float(np.mean(z[minus_idx]))
    return s


@lru_cache(maxsize=1)
def _dimension_ranges():
    art = _art()
    ranges = []
    for dim, plus_idx, minus_idx in _pole_indices():
        scores = []
        for reg in art["registers"].values():
            z = (reg["_feature_mean"] - art["_mean"]) / art["_scale"]
            scores.append(_dim_score(z, plus_idx, minus_idx))
        ranges.append((min(scores), max(scores)))
    return ranges


def dimension_scores(vector):
    """Biber dimension scores for one presence vector.

    Returns [{key, name, left, right, score, lo, hi}] where lo/hi span the
    CORE register profiles, so `score` is interpretable relative to real
    registers (scores outside [lo, hi] are more extreme than any register).
    """
    art = _art()
    z = (np.asarray(vector, dtype=float) - art["_mean"]) / art["_scale"]
    out = []
    for (dim, plus_idx, minus_idx), (lo, hi) in zip(_pole_indices(),
                                                    _dimension_ranges()):
        out.append({"key": dim["key"], "name": dim["name"],
                    "left": dim["left"], "right": dim["right"],
                    "score": _dim_score(z, plus_idx, minus_idx),
                    "lo": lo, "hi": hi})
    return out


def category_profile(probs):
    """Mean presence per feature category, over non-structural features
    (the structural ones are near-constant and would flatten every shape).
    Returns {category: mean}. This is the demo's radar fingerprint."""
    from biberplus.neurobiber.features import groups, structural_codes
    skip = structural_codes()
    p = np.asarray(probs, dtype=float)
    out = {}
    for group in groups():
        idx = [f["index"] for f in feature_list()
               if f["group"] == group and f["code"] not in skip]
        out[group] = round(float(p[idx].mean()), 4) if idx else 0.0
    return out


def fingerprint(probs):
    """Everything the demo's fingerprint tab shows, as one dict:
    category profile, Biber dimension scores, and nearest CORE registers."""
    vec = (np.asarray(probs, dtype=float) > 0.5).astype(float)
    return {
        "categories": category_profile(probs),
        "dimensions": [
            {k: (round(v, 3) if isinstance(v, float) else v)
             for k, v in d.items()}
            for d in dimension_scores(vec)],
        "nearest_registers": [
            {"code": c, "name": n, "similarity": round(s, 3)}
            for c, n, s in nearest_registers(vec)],
    }
