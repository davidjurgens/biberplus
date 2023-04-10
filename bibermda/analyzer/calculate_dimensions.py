from src.analyzer.constants import BIBER_MEANS, BIBER_SDS, TAGS_BY_DIMENSION


def calculate_dimensions(corpus_statistics_df):
    """
    The Dimension scores are calculated using the z-scores of the variables that presented a mean higher
    than 1 in the chart presented in Biber (1988: 77)
    :param corpus_statistics_df: Pandas DataFrame with mean, min, max, range, std for ever linguistic tag
    :return:
    """
    z_scores = calculate_z_scores(corpus_statistics_df)
    dimensions = {}

    for dim_index in range(1, 7):
        dim_index = str(dim_index)
        result = sum(z_scores[tag] for tag in TAGS_BY_DIMENSION[dim_index + '_pos'])
        if dim_index + '_neg' in TAGS_BY_DIMENSION:
            neg_sum = sum(z_scores[tag] for tag in TAGS_BY_DIMENSION[dim_index + '_neg'])
            result -= neg_sum

        dimensions[dim_index] = result

    return dimensions


def calculate_z_scores(corpus_statistics_df):
    z_score_results = {}

    for row in corpus_statistics_df.itertuples():
        if row.tag in BIBER_MEANS:
            mu = BIBER_MEANS[row.tag]
            sigma = BIBER_SDS[row.tag]
            if mu and sigma:
                z_score_results[row.tag] = calculate_z_score(row.mean, mu, sigma)

    return z_score_results


def calculate_z_score(x, mu, sigma):
    return (x - mu) / sigma
