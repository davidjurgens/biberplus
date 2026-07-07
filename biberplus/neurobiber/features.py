"""Metadata for the 96 Neurobiber features: codes, names, descriptions,
Biber Dimension-1 poles, categories.

The order in feature_meta.json mirrors the model's id2label
(Blablablab/neurobiber) and is the canonical index for every 96-dim vector
in this package. No heavy dependencies here.
"""
import json
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def load_meta():
    ref = resources.files("biberplus.neurobiber") / "assets" / "feature_meta.json"
    return json.loads(ref.read_text())


def feature_list():
    """All 96 features, in model output order."""
    return load_meta()["features"]


def feature_codes():
    return [f["code"] for f in feature_list()]


def by_code():
    return {f["code"]: f for f in feature_list()}


def groups():
    """Category names, in display order."""
    return load_meta()["groups"]


def pole_of(code):
    """'involved', 'informational', or 'neutral' (Biber Dimension 1)."""
    return by_code().get(code, {}).get("pole", "neutral")


def name_of(code):
    f = by_code().get(code)
    return f["name"] if f else code


def describe(code):
    f = by_code().get(code)
    return f["description"] if f else code


def structural_codes():
    """Closed-class scaffolding features (nouns, articles, prepositions...)
    that fire on >3% of tokens; hide these when highlighting."""
    return {f["code"] for f in feature_list() if f.get("structural")}
