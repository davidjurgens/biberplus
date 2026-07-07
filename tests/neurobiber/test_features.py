"""Feature metadata: importable with zero neural deps, complete, ordered."""
from biberplus.neurobiber import features


def test_96_features_in_model_order():
    feats = features.feature_list()
    assert len(feats) == 96
    assert feats[0]["code"] == "QUAN"   # id2label[0] == BIN_QUAN
    assert feats[95]["code"] == "X"
    assert [f["index"] for f in feats] == list(range(96))


def test_every_feature_fully_described():
    for f in features.feature_list():
        assert f["name"] and f["description"] and f["group"]
        assert f["pole"] in {"involved", "informational", "neutral"}


def test_pole_spot_checks():
    assert features.pole_of("FPP1") == "involved"
    assert features.pole_of("CONT") == "involved"
    assert features.pole_of("NOMZ") == "informational"
    assert features.pole_of("PASS") == "informational"
    assert features.pole_of("VBD") == "neutral"


def test_structural_codes_are_closed_class():
    s = features.structural_codes()
    assert {"NN", "ART", "PREP", "DET"} <= s
    assert "CONT" not in s and "EMPH" not in s  # the interesting ones stay
