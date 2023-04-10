import numpy as np
from sklearn.metrics import pairwise_distances


def compute_rankings(queries_dimensions, targets, query):
    # Queries = Text from a random author
    # Target = List of authors the queries belong to
    # Query Authors =
    pass


def compute_ranking_metrics(queries, targets, query_authors, target_authors, metric='cosine'):
    num_queries = len(query_authors)
    ranks = np.zeros(num_queries, dtype=np.float32)
    reciprocal_ranks = np.zeros(num_queries, dtype=np.float32)

    distances = pairwise_distances(queries, Y=targets, metric=metric, n_jobs=8)

    for i in range(num_queries):
        dist = distances[i]
        sorted_indices = np.argsort(dist)
        sorted_target_authors = target_authors[sorted_indices]
        ranks[i] = np.where(sorted_target_authors == query_authors[i])[0].item()
        reciprocal_ranks[i] = 1.0 / float(ranks[i] + 1)

    return_dict = {
        'MRR': np.mean(reciprocal_ranks),
        'R@8': np.sum(np.less_equal(ranks + 1, 8)) / np.float32(num_queries),
        'R@50': np.sum(np.less_equal(ranks + 1, 50)) / np.float32(num_queries),
        'R@100': np.sum(np.less_equal(ranks + 1, 100)) / np.float32(num_queries),
        'R@1000': np.sum(np.less_equal(ranks + 1, 1000)) / np.float32(num_queries)
    }

    return return_dict
