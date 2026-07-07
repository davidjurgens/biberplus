"""Authorship interface. Pure-math tests always run; model-dependent tests
run only when an artifact is reachable (NEUROBIBER_AV_PATH or the Hub)."""
import numpy as np
import pytest

pytest.importorskip("sklearn")

from biberplus.neurobiber import authorship


def test_pair_feature_layout():
    a, b = np.zeros(96), np.ones(96)
    x = authorship.pair_features(a, b)
    assert x.shape == (288,)
    assert (x[:96] == 0).all() and (x[96:192] == 1).all() and (x[192:] == 1).all()


def _artifact_reachable():
    try:
        authorship._load()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _artifact_reachable(),
                    reason="authorship artifact not available")
def test_probability_and_drivers():
    rng = np.random.RandomState(0)
    a = rng.randint(0, 2, 96).astype(float)
    p_self = authorship.same_author_probability(a, a)
    assert 0.0 <= p_self <= 1.0
    d = authorship.drivers(a, a, top_n=5)
    assert len(d) == 5 and all(row["agree"] for row in d)
    assert "weighted_f1" in authorship.metrics()
