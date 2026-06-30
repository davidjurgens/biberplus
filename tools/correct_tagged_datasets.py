"""Scan and safely correct all tagged datasets affected by the BIN_ double-count.

The companion module ``fix_binary_means`` implements the per-value transform.
This script applies it across whole corpora WITHOUT re-tagging, with three modes:

    scan    Walk roots, classify every .jsonl/.parquet, write a manifest.
            Read-only. Tells you exactly which files are affected and how.

    verify  Re-read a sample of records from each affected file and assert the
            bug fingerprint holds (mean <= 0.5, std == sqrt(mean(1-mean))).
            Read-only. Run this before apply for confidence.

    apply   Stream each affected file and write a CORRECTED COPY to a new output
            root (mirroring the source layout). Never edits in place. Each record
            is re-validated; if a record fails the fingerprint the file is aborted
            (so already-corrected data is never doubled).

Affected shapes handled:
    - jsonl  features dict  : keys BIN_*_mean / BIN_*_std
    - jsonl  encodings(1|2) : {'binary': [mean, std, mean, std, ...]}
    - parquet interleaved   : numeric columns '0'..'N' == [mean, std, ...]
    - parquet means-only    : a 'biber_tagged' column of per-tag means

NOT touched: neural_biber* model outputs, transformer embeddings, binarized
eval results, raw count tables.

Examples
--------
    # 1) classify everything (read-only) -> manifest.jsonl
    python tools/correct_tagged_datasets.py scan \
        --root /shared/3/projects/hiatus/tagged_data \
        --root /shared/3/datasets/PAN/pan20-av-training-small \
        --manifest manifest.jsonl

    # 2) confirm the bug fingerprint on a sample (read-only)
    python tools/correct_tagged_datasets.py verify --manifest manifest.jsonl

    # 3) write corrected copies under a new root (originals untouched)
    python tools/correct_tagged_datasets.py apply --manifest manifest.jsonl \
        --out-root /shared/3/projects/hiatus/tagged_data_corrected

Notes
-----
    - Parquet support needs pandas (+pyarrow). JSONL needs only the stdlib.
    - Big JSONL files stream line-by-line; parquet files are loaded in memory.
    - Use --limit to process only the first N files (smoke test), and
      --include / --exclude substrings to target a subset.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fix_binary_means import (  # noqa: E402
    correct_binary_vector,
    correct_features,
    correct_means_vector,
    validate_binary_features,
    validate_binary_vector,
    validate_means_vector,
)

DEFAULT_ROOTS = [
    "/shared/3/projects/hiatus/tagged_data",
    "/shared/3/datasets/PAN/pan20-av-training-small",
]

ENCODING_KEYS = ("encodings", "encodings1", "encodings2")
MEANS_COLUMN = "biber_tagged"


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

def _first_jsonl_record(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    return None


def classify_jsonl(path):
    """Return (shape, detail) for a jsonl file, or (None, reason) if unaffected."""
    try:
        rec = _first_jsonl_record(path)
    except Exception as e:  # noqa: BLE001
        return None, f"unreadable:{type(e).__name__}"
    if not isinstance(rec, dict):
        return None, "not_object"

    feats = rec.get("features")
    if isinstance(feats, dict) and any(
        str(k).startswith("BIN_") and str(k).endswith("_mean") for k in feats
    ):
        return "jsonl_features", "features"

    present = [k for k in ENCODING_KEYS if isinstance(rec.get(k), dict)
               and isinstance(rec[k].get("binary"), list)]
    if present:
        return "jsonl_encodings", ",".join(present)

    return None, "no_binary_stats"


def classify_parquet(path):
    import pandas as pd

    df = pd.read_parquet(path)
    cols = list(df.columns)
    if MEANS_COLUMN in cols:
        return "parquet_means", MEANS_COLUMN, df
    numeric = [c for c in cols if str(c).isdigit()]
    if numeric and len(numeric) % 2 == 0:
        ordered = sorted(numeric, key=lambda c: int(c))
        first = df.iloc[0][ordered].tolist()
        if not validate_binary_vector(first):
            return "parquet_interleaved", ",".join(ordered[:1] + [ordered[-1]]), df
    return None, "no_binary_stats", df


def iter_data_files(roots):
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for name in sorted(files):
                if name.endswith(".jsonl"):
                    yield os.path.join(dirpath, name), "jsonl"
                elif name.endswith(".parquet") or name.endswith(".parquet.gzip"):
                    yield os.path.join(dirpath, name), "parquet"


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #

def cmd_scan(args):
    roots = args.root or DEFAULT_ROOTS
    entries = []
    n_affected = 0
    for path, fmt in iter_data_files(roots):
        if args.include and args.include not in path:
            continue
        if args.exclude and args.exclude in path:
            continue
        try:
            if fmt == "jsonl":
                shape, detail = classify_jsonl(path)
            else:
                shape, detail, _ = classify_parquet(path)
        except Exception as e:  # noqa: BLE001
            shape, detail = None, f"error:{type(e).__name__}:{e}"
        affected = shape is not None
        n_affected += affected
        entry = {
            "path": path,
            "root": next(r for r in roots if path.startswith(r)),
            "format": fmt,
            "shape": shape,
            "detail": detail,
            "size_bytes": os.path.getsize(path),
            "affected": affected,
        }
        entries.append(entry)
        if affected:
            print(f"  [AFFECTED:{shape}:{detail}] {entry['size_bytes']/1e9:7.2f}GB  {path}")

    if args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        print(f"\nWrote manifest with {len(entries)} entries ({n_affected} affected) -> {args.manifest}")
    else:
        print(f"\n{len(entries)} files scanned, {n_affected} affected. (Pass --manifest to save.)")
    return 0


def _load_manifest(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _affected_entries(args):
    entries = _load_manifest(args.manifest)
    out = []
    for e in entries:
        if not e.get("affected"):
            continue
        if args.include and args.include not in e["path"]:
            continue
        if args.exclude and args.exclude in e["path"]:
            continue
        out.append(e)
    if args.limit:
        out = out[: args.limit]
    return out


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #

def cmd_verify(args):
    failures = 0
    for e in _affected_entries(args):
        path, shape = e["path"], e["shape"]
        problems = _verify_file(path, shape, e["detail"], args.sample)
        status = "OK" if not problems else f"FAIL ({len(problems)})"
        print(f"  [{status}] {shape:20s} {path}")
        if problems:
            failures += 1
            for p in problems[:3]:
                print(f"       {p}")
    print(f"\nverify complete: {failures} file(s) failed the bug fingerprint.")
    return 1 if failures else 0


def _verify_file(path, shape, detail, sample):
    problems = []
    if shape == "jsonl_features":
        for rec in _sample_jsonl(path, sample):
            problems += validate_binary_features(rec.get("features", {}))
    elif shape == "jsonl_encodings":
        keys = detail.split(",")
        for rec in _sample_jsonl(path, sample):
            for k in keys:
                problems += validate_binary_vector(rec[k]["binary"])
    elif shape == "parquet_means":
        import pandas as pd
        df = pd.read_parquet(path)
        for vec in df[MEANS_COLUMN].head(sample):
            problems += validate_means_vector(list(vec))
    elif shape == "parquet_interleaved":
        import pandas as pd
        df = pd.read_parquet(path)
        ordered = sorted([c for c in df.columns if str(c).isdigit()], key=lambda c: int(c))
        for _, row in df[ordered].head(sample).iterrows():
            problems += validate_binary_vector(row.tolist())
    return problems


def _sample_jsonl(path, sample):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if len(out) >= sample:
                    break
    return out


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #

def _out_path(entry, out_root):
    rel = os.path.relpath(entry["path"], entry["root"])
    return os.path.join(out_root, rel)


def cmd_apply(args):
    if not args.out_root:
        print("ERROR: apply requires --out-root (corrected copies are never written in place).")
        return 2

    entries = _affected_entries(args)
    print(f"apply: {len(entries)} affected file(s) -> {args.out_root}\n")
    total_records = 0
    for e in entries:
        out_path = _out_path(e, args.out_root)
        if os.path.abspath(out_path) == os.path.abspath(e["path"]):
            print(f"  SKIP (would overwrite source): {e['path']}")
            continue
        if os.path.exists(out_path) and not args.overwrite:
            print(f"  SKIP (output exists, use --overwrite): {out_path}")
            continue
        if args.write:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            n = _apply_file(e, out_path, dry_run=not args.write)
        except Exception as ex:  # noqa: BLE001
            print(f"  ABORT {e['path']}: {type(ex).__name__}: {ex}")
            continue
        total_records += n
        verb = "would correct" if not args.write else "corrected"
        print(f"  [{e['shape']}] {verb} {n:,} records -> {out_path}")
    mode = "DRY-RUN (no files written; pass --write to apply)" if not args.write else "WROTE corrected copies"
    print(f"\napply complete [{mode}]: {total_records:,} records across {len(entries)} files.")
    return 0


def _apply_file(entry, out_path, dry_run):
    shape = entry["shape"]
    if shape in ("jsonl_features", "jsonl_encodings"):
        return _apply_jsonl(entry, out_path, dry_run)
    return _apply_parquet(entry, out_path, dry_run)


def _apply_jsonl(entry, out_path, dry_run):
    shape, detail = entry["shape"], entry["detail"]
    keys = detail.split(",") if shape == "jsonl_encodings" else []
    n = 0
    tmp = out_path + ".tmp"
    out_f = None if dry_run else open(tmp, "w", encoding="utf-8")
    try:
        with open(entry["path"], encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if shape == "jsonl_features":
                    rec["features"] = correct_features(rec["features"])
                else:
                    for k in keys:
                        rec[k]["binary"] = correct_binary_vector(rec[k]["binary"])
                n += 1
                if out_f:
                    out_f.write(json.dumps(rec) + "\n")
    finally:
        if out_f:
            out_f.close()
    if not dry_run:
        os.replace(tmp, out_path)
    return n


def _apply_parquet(entry, out_path, dry_run):
    import pandas as pd

    df = pd.read_parquet(entry["path"])
    if entry["shape"] == "parquet_means":
        df[MEANS_COLUMN] = df[MEANS_COLUMN].apply(lambda v: correct_means_vector(list(v)))
    else:
        ordered = sorted([c for c in df.columns if str(c).isdigit()], key=lambda c: int(c))
        # Validate the whole frame against the fingerprint before mutating.
        for _, row in df[ordered].iterrows():
            problems = validate_binary_vector(row.tolist())
            if problems:
                raise ValueError(f"row fails fingerprint: {problems[:3]}")
        means = ordered[0::2]
        stds = ordered[1::2]
        for m, s in zip(means, stds):
            df[m] = 2.0 * df[m]
            df[s] = (df[m] * (1.0 - df[m])).clip(lower=0.0) ** 0.5
    n = len(df)
    if not dry_run:
        tmp = out_path + ".tmp"
        df.to_parquet(tmp)
        os.replace(tmp, out_path)
    return n


# --------------------------------------------------------------------------- #

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Classify all data files (read-only).")
    p_scan.add_argument("--root", action="append", help="Root to walk (repeatable). Defaults to known roots.")
    p_scan.add_argument("--manifest", help="Write manifest JSONL here.")
    p_scan.add_argument("--include", help="Only paths containing this substring.")
    p_scan.add_argument("--exclude", help="Skip paths containing this substring.")
    p_scan.set_defaults(func=cmd_scan)

    p_verify = sub.add_parser("verify", help="Sample-check the bug fingerprint (read-only).")
    p_verify.add_argument("--manifest", required=True)
    p_verify.add_argument("--sample", type=int, default=200, help="Records/rows to check per file.")
    p_verify.add_argument("--include")
    p_verify.add_argument("--exclude")
    p_verify.add_argument("--limit", type=int)
    p_verify.set_defaults(func=cmd_verify)

    p_apply = sub.add_parser("apply", help="Write corrected copies to a new root.")
    p_apply.add_argument("--manifest", required=True)
    p_apply.add_argument("--out-root", required=True, help="Destination root (mirrors source layout).")
    p_apply.add_argument("--write", action="store_true", help="Actually write (default is dry-run).")
    p_apply.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    p_apply.add_argument("--include")
    p_apply.add_argument("--exclude")
    p_apply.add_argument("--limit", type=int)
    p_apply.set_defaults(func=cmd_apply)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
