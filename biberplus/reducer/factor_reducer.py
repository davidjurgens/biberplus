import pandas as pd
from factor_analyzer import FactorAnalyzer


def tags_factor_analysis(frequencies_df, n_factors=3, name=None, rotation="promax"):
    fa = FactorAnalyzer(rotation=rotation, n_factors=n_factors)
    X = frequencies_df.drop("tag", axis=1).values
    fa.fit(X)

    columns = ["Factor{}".format(i + 1) for i in range(n_factors)]
    loadings_df = pd.DataFrame(data=fa.loadings_, columns=columns)

    if name:
        loadings_df["name"] = name

    return loadings_df, fa
