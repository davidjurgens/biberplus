"""Measure Neurobiber's tagging rate on this machine and print it next to
the paper's reference rates. Needs biberplus[neural].

    python examples/throughput.py
"""
import time

from biberplus.neurobiber.model import get_device, load_model, predict_probs

words = ("The committee reviewed the proposal and, honestly, we didn't "
         "think it would pass. ") * 64
texts = [words] * 40
total = sum(len(t.split()) for t in texts)

load_model()  # exclude cold start
t0 = time.perf_counter()
predict_probs(texts)
elapsed = time.perf_counter() - t0

print(f"{total / elapsed:,.0f} tokens/s on {get_device()} "
      f"({total:,} tokens in {elapsed:.1f}s)")
print("Paper reference: 29,095 tok/s (A6000) · 14,076 (16-core rules) · "
      "2,607 (1-core rules) · 1,185 (MAT)")
