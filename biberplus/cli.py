"""The biberplus / neurobiber command line.

    biberplus tag text.txt            # rule tagger: per-token fired features
    biberplus vector text.txt         # neural: named 96-dim probabilities
    biberplus fingerprint text.txt    # categories + dimensions + registers
    biberplus compare a.txt b.txt     # same-author probability + drivers
    biberplus benchmark               # measured tokens/s on this machine

Every text argument may be a file path or `-` for stdin. `vector` and
`fingerprint` also accept `--jsonl corpus.jsonl --text-key text` for batch
tagging (one JSON object per line on stdout). Neural commands need
`pip install biberplus[neural]`; `tag` runs on the rule tagger alone.
"""
import argparse
import json
import sys
import time


def _read_text(path):
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8") as f:
        return f.read()


def _iter_jsonl(path, text_key):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _require_neural(what):
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        sys.exit(f"`{what}` needs the neural extra: pip install biberplus[neural]")


# ---------------------------------------------------------------- tag

def cmd_tag(args):
    from biberplus.tagger.biber_plus_tagger import BiberPlusTagger
    from biberplus.tagger.tagger import word2dict
    from biberplus.tagger.tagger_utils import (build_variable_dictionaries,
                                               load_config, load_pipeline)
    from biberplus.neurobiber.features import (feature_codes, name_of,
                                               structural_codes)

    text = _read_text(args.text)
    config = load_config()
    config["function_words"] = False
    doc = load_pipeline(config)(text)
    tagged = [word2dict(t) for t in doc]
    if tagged:
        tagged = BiberPlusTagger(tagged, build_variable_dictionaries()).run_all()

    known = set(feature_codes())
    skip = set() if args.all else structural_codes()
    rows = [(w["text"], [t for t in w["tags"] if t in known and t not in skip])
            for w in tagged]

    if args.json:
        print(json.dumps([{"token": tok, "features": tags}
                          for tok, tags in rows]))
        return
    for tok, tags in rows:
        if tags:
            print(f"{tok:20s} {', '.join(f'{t} ({name_of(t)})' for t in tags)}")
    n = sum(1 for _, tags in rows if tags)
    hidden = "" if args.all else " (structural features hidden; --all shows them)"
    print(f"\n{n}/{len(rows)} tokens fired{hidden}", file=sys.stderr)


# ---------------------------------------------------------------- vector

def cmd_vector(args):
    _require_neural("vector")
    from biberplus.neurobiber.model import named_vector, predict_probs
    from biberplus.neurobiber.features import feature_codes

    if args.jsonl:
        codes = feature_codes()
        for item in _iter_jsonl(args.jsonl, args.text_key):
            probs = predict_probs([item.get(args.text_key, "")])[0]
            item["neurobiber"] = {c: round(float(p), 4)
                                  for c, p in zip(codes, probs)}
            print(json.dumps(item))
        return
    print(json.dumps(named_vector(_read_text(args.text)), indent=None))


# ---------------------------------------------------------------- fingerprint

def _print_fingerprint(fp, label=""):
    if label:
        print(f"== {label}")
    print("Categories (mean presence, non-structural):")
    for cat, v in sorted(fp["categories"].items(), key=lambda x: -x[1]):
        bar = "#" * int(round(20 * v))
        print(f"  {cat:36s} {v:.2f} {bar}")
    print("Biber dimensions (relative to CORE registers):")
    for d in fp["dimensions"]:
        lo, hi, s = d["lo"], d["hi"], d["score"]
        frac = min(max((s - lo) / ((hi - lo) or 1), 0), 1)
        pos = int(round(frac * 20))
        track = "-" * pos + "o" + "-" * (20 - pos)
        print(f"  {d['right']:>22s} {track} {d['left']:22s} ({s:+.2f})")
    regs = " · ".join(r["name"] for r in fp["nearest_registers"])
    print(f"Reads most like: {regs}")


def cmd_fingerprint(args):
    _require_neural("fingerprint")
    from biberplus.neurobiber.model import predict_probs
    from biberplus.neurobiber import fingerprint as fp_mod

    if args.jsonl:
        for item in _iter_jsonl(args.jsonl, args.text_key):
            probs = predict_probs([item.get(args.text_key, "")],
                                  max_chunks_per_text=1)[0]
            item["fingerprint"] = fp_mod.fingerprint(probs)
            print(json.dumps(item))
        return

    texts = [(args.text, "Text A")] + ([(args.pair, "Text B")] if args.pair else [])
    results = []
    for path, label in texts:
        # first chunk only: matches the artifact's presence window
        probs = predict_probs([_read_text(path)], max_chunks_per_text=1)[0]
        results.append((label, fp_mod.fingerprint(probs)))
    if args.json:
        print(json.dumps({label: fp for label, fp in results}))
        return
    for label, fp in results:
        _print_fingerprint(fp, label if len(results) > 1 else "")


# ---------------------------------------------------------------- compare

def cmd_compare(args):
    _require_neural("compare")
    from biberplus.neurobiber.authorship import compare_texts
    res = compare_texts(_read_text(args.text_a), _read_text(args.text_b))
    if args.json:
        print(json.dumps(res))
        return
    p = res["same_author_probability"]
    print(f"Same-author probability: {p:.2f} -> {res['verdict']} "
          f"(threshold {res['threshold']})")
    print("Most influential features:")
    for d in res["drivers"]:
        mark = "matches" if d["agree"] else "differs"
        print(f"  {d['code']:7s} {d['name']:28s} "
              f"A={'present' if d['a'] else 'absent':7s} "
              f"B={'present' if d['b'] else 'absent':7s} {mark}")
    m = res["model_metrics"]
    if m:
        print(f"Model: weighted F1 {m.get('weighted_f1', float('nan')):.2f} "
              f"on its PAN 2020 split. {res['caveat']}")


# ---------------------------------------------------------------- benchmark

REFERENCE_RATES = [
    ("Neurobiber, 1 GPU (paper, A6000)", 29095),
    ("BiberPlus, 16 cores (paper)", 14076),
    ("BiberPlus, 1 core (paper)", 2607),
    ("MAT, 1 core (paper)", 1185),
]


def cmd_benchmark(args):
    _require_neural("benchmark")
    from biberplus.neurobiber.model import get_device, load_model, predict_probs

    words = ("The committee reviewed the proposal and, honestly, "
             "we didn't think it would pass. ") * 64
    texts = [words] * max(1, args.tokens // len(words.split()))
    total = sum(len(t.split()) for t in texts)

    load_model()  # exclude cold start from the measured rate
    t0 = time.perf_counter()
    predict_probs(texts, subbatch_size=args.batch)
    elapsed = time.perf_counter() - t0
    rate = total / elapsed

    print(f"Measured: {rate:,.0f} tokens/s on {get_device()} "
          f"({total:,} tokens in {elapsed:.1f}s)")
    print("Reference rates (end-to-end, from the paper):")
    for label, ref in REFERENCE_RATES:
        print(f"  {label:36s} {ref:>7,} tokens/s")


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="biberplus",
        description="Interpretable stylistic analysis: 96 Biber features, "
                    "rule-based or neural.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tag", help="rule tagger: which features fire on each token")
    p.add_argument("text", help="file path or - for stdin")
    p.add_argument("--all", action="store_true",
                   help="include structural features (nouns, articles...)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_tag)

    p = sub.add_parser("vector", help="neural: named 96-dim probability vector")
    p.add_argument("text", nargs="?", default="-")
    p.add_argument("--jsonl", help="batch: jsonl file, one object per line")
    p.add_argument("--text-key", default="text")
    p.set_defaults(func=cmd_vector)

    p = sub.add_parser("fingerprint",
                       help="category profile + Biber dimensions + registers")
    p.add_argument("text", nargs="?", default="-")
    p.add_argument("--pair", help="second text for an A/B contrast")
    p.add_argument("--jsonl", help="batch: jsonl file")
    p.add_argument("--text-key", default="text")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_fingerprint)

    p = sub.add_parser("compare", help="same-author probability for two texts")
    p.add_argument("text_a")
    p.add_argument("text_b")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("benchmark", help="measure tokens/s on this machine")
    p.add_argument("--tokens", type=int, default=50_000)
    p.add_argument("--batch", type=int, default=16)
    p.set_defaults(func=cmd_benchmark)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
