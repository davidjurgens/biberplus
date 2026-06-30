"""Post-hoc correction for the BIN_ double-count bug.

Older biberplus (<= 0.4.0) halved every binary feature mean (and produced an
inflated std) because ``BIN_`` tags were counted twice per window: once as the
real presence indicator and once as a spurious ``0``. The corruption is a
deterministic, invertible transform on the stored aggregate statistics, so data
that was already tagged can be corrected WITHOUT re-running the tagger:

    corrected_mean = 2 * stored_mean
    corrected_std  = sqrt(p * (1 - p))      with p = corrected_mean

(The stored std is not needed for recovery; for any 0/1-derived series the std
is fully determined by the mean.)

Bug fingerprint / validation invariants (a record must satisfy these to be a
genuine product of this bug, otherwise we refuse to transform it):

    - every BIN_*_mean <= 0.5
    - every BIN_*_std == sqrt(stored_mean * (1 - stored_mean))

This module exposes pure helpers (reusable for JSONL / CSV / parquet rows or for
flattened ``encode_text`` binary vectors) plus a dry-run-first CLI. It NEVER
mutates a file in place: corrected output is always written to a new path.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Sequence, Tuple

EPS = 1e-6
STD_TOL = 1e-4
MEAN_SUFFIX = "_mean"
STD_SUFFIX = "_std"
BIN_PREFIX = "BIN_"


def _is_bin_mean_key(key: str) -> bool:
    return key.startswith(BIN_PREFIX) and key.endswith(MEAN_SUFFIX)


def _is_bin_std_key(key: str) -> bool:
    return key.startswith(BIN_PREFIX) and key.endswith(STD_SUFFIX)


def corrected_mean(stored_mean: float) -> float:
    return 2.0 * stored_mean


def corrected_std(corr_mean: float) -> float:
    return math.sqrt(max(corr_mean * (1.0 - corr_mean), 0.0))


def expected_buggy_std(stored_mean: float) -> float:
    """Std a 0/1-derived series with fraction-of-ones == stored_mean must have."""
    return math.sqrt(max(stored_mean * (1.0 - stored_mean), 0.0))


def validate_binary_features(features: Dict[str, float], eps: float = EPS) -> List[Tuple[str, str]]:
    """Return a list of (key, reason) invariant violations. Empty == matches bug."""
    problems: List[Tuple[str, str]] = []
    for key, val in features.items():
        if _is_bin_mean_key(key):
            if val > 0.5 + eps:
                problems.append((key, f"mean {val} > 0.5"))
            std_key = key[: -len(MEAN_SUFFIX)] + STD_SUFFIX
            if std_key in features:
                expected = expected_buggy_std(val)
                if abs(features[std_key] - expected) > STD_TOL:
                    problems.append(
                        (std_key, f"std {features[std_key]} != sqrt(mean(1-mean))={expected}")
                    )
    return problems


def correct_features(features: Dict[str, float], validate: bool = True, eps: float = EPS) -> Dict[str, float]:
    """Return a corrected copy of a features dict (BIN_*_mean doubled, std recomputed)."""
    if validate:
        problems = validate_binary_features(features, eps)
        if problems:
            raise ValueError(f"record fails bug-fingerprint invariants: {problems[:3]}")

    corrected = dict(features)
    for key, val in features.items():
        if _is_bin_mean_key(key):
            cm = corrected_mean(val)
            corrected[key] = cm
            std_key = key[: -len(MEAN_SUFFIX)] + STD_SUFFIX
            if std_key in corrected:
                corrected[std_key] = corrected_std(cm)
    return corrected


def validate_binary_vector(vec: Sequence[float], eps: float = EPS) -> List[Tuple[int, str]]:
    """Validate a flattened [mean0, std0, mean1, std1, ...] encode_text binary block."""
    problems: List[Tuple[int, str]] = []
    for i in range(0, len(vec) - 1, 2):
        mean, std = vec[i], vec[i + 1]
        if mean > 0.5 + eps:
            problems.append((i, f"mean {mean} > 0.5"))
        expected = expected_buggy_std(mean)
        if abs(std - expected) > STD_TOL:
            problems.append((i + 1, f"std {std} != sqrt(mean(1-mean))={expected}"))
    return problems


def correct_binary_vector(vec: Sequence[float], validate: bool = True, eps: float = EPS) -> List[float]:
    """Correct a flattened [mean, std, mean, std, ...] encode_text binary block."""
    if validate:
        problems = validate_binary_vector(vec, eps)
        if problems:
            raise ValueError(f"vector fails bug-fingerprint invariants: {problems[:3]}")

    out = list(vec)
    for i in range(0, len(out) - 1, 2):
        cm = corrected_mean(out[i])
        out[i] = cm
        out[i + 1] = corrected_std(cm)
    return out


def validate_means_vector(vec: Sequence[float], eps: float = EPS) -> List[Tuple[int, str]]:
    """Validate a means-only binary vector (one BIN_*_mean per position, no std)."""
    return [(i, f"mean {v} > 0.5") for i, v in enumerate(vec) if v > 0.5 + eps]


def correct_means_vector(vec: Sequence[float], validate: bool = True, eps: float = EPS) -> List[float]:
    """Correct a means-only binary vector (each value is a BIN_*_mean -> doubled).

    Used for stores that keep only the per-tag presence fraction (no std), e.g. a
    96-length ``biber_tagged`` vector.
    """
    if validate:
        problems = validate_means_vector(vec, eps)
        if problems:
            raise ValueError(f"means vector fails bug-fingerprint invariants: {problems[:3]}")
    return [corrected_mean(v) for v in vec]


def _correct_record(record: dict, validate: bool) -> Tuple[dict, int]:
    """Correct a JSONL record in place-ish (returns a new dict + count of fields changed)."""
    changed = 0

    if isinstance(record.get("features"), dict):
        before = record["features"]
        record = dict(record)
        record["features"] = correct_features(before, validate=validate)
        changed += sum(1 for k in before if _is_bin_mean_key(k) or _is_bin_std_key(k))

    for enc_key in ("encodings", "encodings1", "encodings2"):
        enc = record.get(enc_key)
        if isinstance(enc, dict) and isinstance(enc.get("binary"), list):
            record = dict(record)
            new_enc = dict(enc)
            new_enc["binary"] = correct_binary_vector(enc["binary"], validate=validate)
            record[enc_key] = new_enc
            changed += len(enc["binary"])

    return record, changed


def process_jsonl(input_path: str, output_path: str | None, validate: bool = True) -> dict:
    """Validate (and optionally write a corrected copy of) a JSONL artifact.

    When output_path is None this is a dry run: nothing is written, only a report
    is returned. Output is always a NEW file; in-place edits are not supported.
    """
    if output_path is not None and output_path == input_path:
        raise ValueError("Refusing to write in place; choose a different output_path.")

    report = {"records": 0, "records_changed": 0, "fields_changed": 0, "violations": []}
    out_f = open(output_path, "w", encoding="utf-8") if output_path else None
    try:
        with open(input_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                report["records"] += 1
                try:
                    corrected, changed = _correct_record(record, validate=validate)
                except ValueError as e:
                    report["violations"].append((line_no, str(e)))
                    if out_f:
                        out_f.close()
                    raise
                if changed:
                    report["records_changed"] += 1
                    report["fields_changed"] += changed
                if out_f:
                    out_f.write(json.dumps(corrected) + "\n")
    finally:
        if out_f:
            out_f.close()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Input JSONL artifact produced by the buggy tagger.")
    parser.add_argument(
        "--out",
        default=None,
        help="Write a corrected copy to this NEW path. Omit for a dry run (validate + report only).",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the bug-fingerprint invariant check (NOT recommended).",
    )
    args = parser.parse_args(argv)

    report = process_jsonl(args.input, args.out, validate=not args.no_validate)
    mode = "WROTE" if args.out else "DRY-RUN"
    print(f"[{mode}] {args.input}")
    print(f"  records:         {report['records']:,}")
    print(f"  records changed: {report['records_changed']:,}")
    print(f"  fields changed:  {report['fields_changed']:,}")
    if args.out:
        print(f"  output:          {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
