"""A text's style fingerprint: category profile, Biber dimension scores,
and the CORE registers it reads most like. Needs biberplus[neural].

    python examples/style_fingerprint.py
"""
from biberplus.neurobiber.model import predict_probs
from biberplus.neurobiber.fingerprint import fingerprint

TEXT = (
    "The extraction of interpretable stylistic features at scale remains a "
    "significant limitation of existing computational approaches. This "
    "investigation presents a systematic evaluation of the relationship "
    "between register variation and grammatical features in web corpora."
)

# first chunk only: matches the fitted artifact's presence window
probs = predict_probs([TEXT], max_chunks_per_text=1)[0]
fp = fingerprint(probs)

print("Top categories:")
for cat, v in sorted(fp["categories"].items(), key=lambda x: -x[1])[:5]:
    print(f"  {cat:36s} {v:.2f}")

print("Biber dimensions:")
for d in fp["dimensions"]:
    side = d["left"] if d["score"] > (d["lo"] + d["hi"]) / 2 else d["right"]
    print(f"  {d['name']:44s} -> {side} ({d['score']:+.2f})")

print("Reads most like:",
      " · ".join(r["name"] for r in fp["nearest_registers"]))
