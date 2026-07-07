"""Convert a fitted CORE PCA pickle (from the demo's build_pca.py) into the
version-safe JSON artifact shipped as biberplus/neurobiber/assets/core_fingerprint.json.

JSON instead of pickle: no sklearn version coupling, no pickle security
caveats, small enough for the wheel (~500 KB).

Usage:
  python scripts/export_core_fingerprint.py /path/to/pca_core.pkl
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "biberplus" / "neurobiber" / "assets" / "core_fingerprint.json"


def rnd(arr, nd=5):
    return np.round(np.asarray(arr, dtype=float), nd).tolist()


def main():
    with open(sys.argv[1], "rb") as f:
        art = pickle.load(f)

    registers = {}
    for code, centroid in art["register_centroids"].items():
        registers[code] = {
            "name": art["register_names"].get(code, code),
            "count": int(art["register_counts"].get(code, 0)),
            "centroid": rnd(centroid),
            "feature_mean": rnd(art["register_feature_means"][code]),
        }

    out = {
        "source": f"CORE corpus ({art.get('source', 'dev split')}), "
                  "first 512-token chunk presence, "
                  f"{art.get('n_docs', '?')} documents",
        "feature_codes": art["feature_codes"],
        "scaler_mean": rnd(art["scaler"].mean_),
        "scaler_scale": rnd(art["scaler"].scale_),
        "components": rnd(art["pca"].components_),
        "explained_variance_ratio": rnd(art["pca"].explained_variance_ratio_, 6),
        "dim1_pc": int(art["dim1_pc"]),
        "dim1_sign": int(art["dim1_sign"]),
        "registers": registers,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, "
          f"{len(registers)} registers)")


if __name__ == "__main__":
    main()
