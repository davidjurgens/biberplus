"""CLI: parseable without neural deps; `tag` end-to-end; neural verbs
end-to-end when torch is present."""
import json
import subprocess
import sys

import pytest


def run_cli(args, stdin=None):
    return subprocess.run([sys.executable, "-m", "biberplus.cli"] + args,
                          input=stdin, capture_output=True, text=True,
                          timeout=600)


def test_help_needs_no_neural_deps():
    out = run_cli(["--help"])
    assert out.returncode == 0
    for verb in ["tag", "vector", "fingerprint", "compare", "benchmark"]:
        assert verb in out.stdout


def test_tag_stdin_json():
    out = run_cli(["tag", "-", "--json"],
                  stdin="It doesn't seem likely that we'll finish.")
    assert out.returncode == 0, out.stderr
    rows = json.loads(out.stdout)
    tokens = {r["token"]: r["features"] for r in rows}
    assert "PIT" in tokens["It"]
    assert any("XX0" in f for f in tokens.values())


def test_vector_stdin():
    pytest.importorskip("torch")
    out = run_cli(["vector", "-"], stdin="It doesn't seem likely!")
    assert out.returncode == 0, out.stderr
    vec = json.loads(out.stdout)
    assert len(vec) == 96 and vec["PIT"] > 0.5


def test_fingerprint_stdin_json():
    pytest.importorskip("torch")
    out = run_cli(["fingerprint", "-", "--json"],
                  stdin="The committee's evaluation was conducted under "
                        "the established framework and recorded in 1989.")
    assert out.returncode == 0, out.stderr
    fp = json.loads(out.stdout)["Text A"]
    assert set(fp) == {"categories", "dimensions", "nearest_registers"}
