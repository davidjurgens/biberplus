import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from losses import *


def compute_ranking_metrics(queries, targets, query_authors, target_authors, metric='cosine'):
    num_queries = len(query_authors)
    print("Computing ranking metrics for {} queries".format(num_queries))
    ranks = np.zeros((num_queries), dtype=np.float32)
    reciprocal_ranks = np.zeros((num_queries), dtype=np.float32)
    distances = pairwise_distances(queries, Y=targets, metric=metric, n_jobs=6)

    errors = []
    for i in range(num_queries):
        try:
            dist = distances[i]
            sorted_indices = np.argsort(dist)
            sorted_target_authors = target_authors[sorted_indices]
            ranks[i] = np.where(sorted_target_authors == query_authors[i])[0].item()
            reciprocal_ranks[i] = 1.0 / float(ranks[i] + 1)
        except:
            errors.append(i)
    ranks[errors] = -100
    reciprocal_ranks[errors] = -100
    ranks = ranks[ranks != -100]
    reciprocal_ranks = reciprocal_ranks[reciprocal_ranks != -100]

    return_dict = {
        'MRR': np.mean(reciprocal_ranks),
        'R@8': np.sum(np.less_equal(ranks + 1, 8)) / np.float32(num_queries),
        'R@50': np.sum(np.less_equal(ranks + 1, 50)) / np.float32(num_queries),
        'R@100': np.sum(np.less_equal(ranks + 1, 100)) / np.float32(num_queries),
        'R@1000': np.sum(np.less_equal(ranks + 1, 1000)) / np.float32(num_queries)
    }

    return return_dict


def evaluate(model, dataloader, args):
    device = args.device

    model.eval()
    model.to(device)
    print('Evaluating...')
    queries, targets = [], []
    all_query_authors, all_target_authors = [], []

    with torch.autocast(device_type="cuda", enabled=True):
        for batchA, batchB, _, _, query_authors, target_authors in tqdm(dataloader):
            with torch.no_grad():
                query = model(**batchA.to(device)).pooler_output
                target = model(**batchB.to(device)).pooler_output
                queries.append(query.cpu().detach().numpy())
                targets.append(target.cpu().detach().numpy())
                all_query_authors += query_authors
                all_target_authors += target_authors

    all_target_authors = np.array(all_target_authors)
    all_query_authors = np.array(all_query_authors)
    queries = np.concatenate(queries, axis=0)
    targets = np.concatenate(targets, axis=0)

    return compute_ranking_metrics(queries, targets, all_query_authors, all_target_authors, args.metric)
