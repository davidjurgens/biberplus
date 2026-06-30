"""Shape tests for the dimensionality-reduction helpers."""
import unittest

import numpy as np
import pandas as pd

from biberplus.reducer.pca_reducer import tags_pca

try:
    from biberplus.reducer.factor_reducer import tags_factor_analysis
    _HAS_FACTOR = True
except ImportError:
    _HAS_FACTOR = False


def _frequencies_df(n_rows=30, seed=0):
    rng = np.random.default_rng(seed)
    data = {
        "tag": [f"T{i}" for i in range(n_rows)],
        "mean": rng.random(n_rows),
        "min_val": rng.random(n_rows),
        "max_val": rng.random(n_rows),
        "range": rng.random(n_rows),
        "std": rng.random(n_rows),
    }
    return pd.DataFrame(data)


class TestTagsPca(unittest.TestCase):
    def test_output_shape_and_variance(self):
        df = _frequencies_df()
        pca_df, explained_variance = tags_pca(df, components=2)
        self.assertEqual(pca_df.shape, (len(df), 2))
        self.assertEqual(list(pca_df.columns), ["PC1", "PC2"])
        self.assertEqual(len(explained_variance), 2)

    def test_name_column_added(self):
        df = _frequencies_df()
        pca_df, _ = tags_pca(df, components=3, name="sample")
        self.assertIn("name", pca_df.columns)
        self.assertTrue((pca_df["name"] == "sample").all())


@unittest.skipUnless(_HAS_FACTOR, "factor_analyzer not installed")
class TestTagsFactorAnalysis(unittest.TestCase):
    def test_loadings_shape(self):
        df = _frequencies_df()
        try:
            loadings_df, fa = tags_factor_analysis(df, n_factors=2)
        except TypeError as e:
            # factor_analyzer <= 0.5 calls sklearn check_array(force_all_finite=...),
            # which was removed in sklearn >= 1.6. Pre-existing env incompatibility,
            # unrelated to this change set.
            if "force_all_finite" in str(e):
                self.skipTest(f"factor_analyzer incompatible with installed sklearn: {e}")
            raise
        # loadings_ is (n_features, n_factors); features are the 5 value columns.
        self.assertEqual(loadings_df.shape, (5, 2))
        self.assertEqual(list(loadings_df.columns), ["Factor1", "Factor2"])


if __name__ == "__main__":
    unittest.main()
