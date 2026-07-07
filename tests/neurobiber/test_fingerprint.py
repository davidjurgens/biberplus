"""Fingerprint math from the shipped JSON artifact — numpy only."""
import numpy as np
import pytest

from biberplus.neurobiber import fingerprint as fp
from biberplus.neurobiber.features import feature_codes, pole_of


def _vec(codes_on):
    codes = feature_codes()
    v = np.zeros(96)
    for c in codes_on:
        v[codes.index(c)] = 1.0
    return v


def test_artifact_loads_and_matches_feature_order():
    assert "CORE" in fp.source()
    assert fp.explained_variance_ratio().shape == (96,)
    assert len(fp.register_map()) >= 20


def test_project_shape():
    assert fp.project(np.zeros((3, 96))).shape == (3, 96)


def test_dim1_loadings_align_with_poles():
    pos, neg = fp.loadings(fp._art()["dim1_pc"], top_n=6)
    sign = fp._art()["dim1_sign"]
    hi, lo = (pos, neg) if sign > 0 else (neg, pos)
    assert sum(pole_of(c) == "involved" for c, _ in hi) >= 2
    assert sum(pole_of(c) == "informational" for c, _ in lo) >= 2


def test_dimension_scores_shape_and_ranges():
    rows = fp.dimension_scores(np.zeros(96))
    assert [r["key"] for r in rows] == ["d1", "d2", "d3", "d4", "d5"]
    for r in rows:
        assert r["lo"] < r["hi"]


def test_involved_vector_scores_involved():
    involved = _vec(["FPP1", "SPP2", "CONT", "EMPH", "UH", "EXCL", "QUES",
                     "PIT", "DEMP", "XX0", "HDG", "AMP"])
    informational = _vec(["NOMZ", "PASS", "BYPA", "PIN", "JJ", "WZPAST",
                          "TSUB", "PIRE", "THVC", "GER", "NNP", "ART"])
    d1_inv = fp.dimension_scores(involved)[0]["score"]
    d1_inf = fp.dimension_scores(informational)[0]["score"]
    assert d1_inv > d1_inf


def test_narrative_vector_scores_narrative():
    narrative = _vec(["VBD", "TPP3", "PEAS", "PUBV", "PRESP", "WZPAST"])
    d2 = fp.dimension_scores(narrative)[1]
    assert d2["score"] > 0


def test_nearest_registers_returns_ranked_names():
    out = fp.nearest_registers(np.ones(96), k=4)
    assert len(out) == 4
    sims = [s for _, _, s in out]
    assert sims == sorted(sims, reverse=True)


def test_category_profile_covers_all_groups():
    from biberplus.neurobiber.features import groups
    prof = fp.category_profile(np.ones(96) * 0.5)
    assert set(prof) == set(groups())
    assert all(0 <= v <= 1 for v in prof.values())


def test_fingerprint_bundle_shape():
    out = fp.fingerprint(np.random.RandomState(0).rand(96))
    assert set(out) == {"categories", "dimensions", "nearest_registers"}
    assert len(out["nearest_registers"]) == 3
