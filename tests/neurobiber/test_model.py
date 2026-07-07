"""Neural model tests. Chunking is pure-python; inference tests download
the model and are skipped when torch/transformers are unavailable."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biberplus.neurobiber.model import (chunk_text, named_vector,
                                        predict_binary, predict_probs)
from biberplus.neurobiber.features import feature_codes

IDX = {c: i for i, c in enumerate(feature_codes())}


def test_chunking_512_whitespace_tokens():
    text = " ".join(f"w{i}" for i in range(1200))
    chunks = chunk_text(text)
    assert len(chunks) == 3
    assert len(chunks[0].split()) == 512
    assert len(chunks[2].split()) == 176


def test_chunking_empty():
    assert chunk_text("   ") == []


def test_empty_text_gives_zero_vector():
    assert predict_probs(["   "]).sum() == 0


def test_paper_example_features_fire():
    # Figure 1 of the paper: PIT, CONT, XX0, SMP must all fire
    vec = predict_binary(["It doesn't seem likely that we'll finish today."])[0]
    for code in ["PIT", "CONT", "XX0", "SMP"]:
        assert vec[IDX[code]] == 1, f"{code} should fire"


def test_long_document_span_or():
    """Features present only in the SECOND 512-token span survive the
    max-across-chunks aggregation."""
    filler = " ".join(["The report was reviewed by the committee ."] * 80)
    tail = " ".join(["Wow!!! I don't think it's likely, lol :) #style @you"] * 20)
    doc = filler + " " + tail

    tail_vec = predict_binary([tail])[0]
    doc_vec = predict_binary([doc])[0]
    fired = [c for c in ["EMOT", "HASH", "AT", "LAUGH", "EXCL", "CONT"]
             if tail_vec[IDX[c]] == 1]
    assert len(fired) >= 4
    lost = [c for c in fired if doc_vec[IDX[c]] == 0]
    # chunk-level predictions are context-sensitive; allow at most one miss
    assert len(lost) <= 1, f"span aggregation lost too much: {lost}"


def test_max_chunks_cap():
    filler = " ".join(["The report was reviewed by the committee ."] * 80)
    tail = " ".join(["Wow!!! lol #tag @you"] * 30)
    doc = filler + " " + tail
    full = predict_binary([doc])[0]
    capped = predict_binary([doc], max_chunks_per_text=1)[0]
    tail_only = [c for c in ["EMOT", "HASH", "AT", "LAUGH"]
                 if full[IDX[c]] == 1]
    assert tail_only
    assert all(capped[IDX[c]] == 0 for c in tail_only)


def test_named_vector_is_json_friendly():
    import json
    nv = named_vector("It doesn't seem likely.")
    assert len(nv) == 96
    json.dumps(nv)
    assert nv["PIT"] > 0.5


def test_agreement_with_rule_tagger():
    """Neurobiber reproduces BiberPlus presence labels on short texts
    (paper: macro-F1 0.97; per-text agreement is the small-sample proxy)."""
    from biberplus.tagger.biber_plus_tagger import BiberPlusTagger
    from biberplus.tagger.tagger import word2dict
    from biberplus.tagger.tagger_utils import (build_variable_dictionaries,
                                               load_config, load_pipeline)
    texts = [
        "It doesn't seem likely that we'll finish today, but I hope we do!",
        "The committee's evaluation was conducted under the framework.",
        "ok so this new coffee place?? absolutely unreal. don't walk, RUN.",
    ]
    config = load_config()
    config["function_words"] = False
    nlp = load_pipeline(config)
    patterns = build_variable_dictionaries()
    codes = feature_codes()

    agreements = []
    for text in texts:
        tagged = BiberPlusTagger([word2dict(t) for t in nlp(text)],
                                 patterns).run_all()
        fired = {t for w in tagged for t in w["tags"]}
        gold = np.array([1 if c in fired else 0 for c in codes])
        pred = predict_binary([text])[0]
        agreements.append((gold == pred).mean())
    assert np.mean(agreements) > 0.90, agreements
