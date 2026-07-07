"""Neurobiber: neural tagging of the 96 Biber features, plus the analyses
from the toolkit's live demo — style fingerprints, register placement,
Biber dimension scores, and authorship comparison.

The neural pieces need the optional extra:

    pip install biberplus[neural]

`biberplus.neurobiber.features` and `fingerprint` work without torch;
`model` (tagging) and `authorship` (comparison) require it.
"""
from biberplus.neurobiber import features  # noqa: F401  (always importable)

__all__ = ["features", "fingerprint", "model", "authorship"]


def __getattr__(name):
    # Lazy imports so `import biberplus` never drags in torch, and users
    # without the extra get a clear message only when they touch it.
    if name in ("model", "authorship", "fingerprint"):
        import importlib
        try:
            return importlib.import_module(f"biberplus.neurobiber.{name}")
        except ImportError as e:
            raise ImportError(
                f"biberplus.neurobiber.{name} needs the neural extra: "
                "pip install biberplus[neural]"
            ) from e
    raise AttributeError(name)
