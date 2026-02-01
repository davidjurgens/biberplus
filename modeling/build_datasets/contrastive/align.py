import pandas as pd


def align(candidate_path, query_path):
    candidates = pd.read_json(candidate_path, orient="records", lines=True)
    queries = pd.read_json(query_path, orient="records", lines=True)

    candidates["authorIDs"] = candidates["authorIDs"].astype("str")
    queries["authorIDs"] = queries["authorIDs"].astype("str")

    candidates = candidates.sort_values(["authorIDs"])
    queries = queries.sort_values(["authorIDs"])

    assert (
        set(queries["documentID"]).intersection(set(candidates["documentID"])) == set()
    )
    assert queries["authorIDs"].tolist() == candidates["authorIDs"].tolist()

    candidates.to_json(candidate_path, orient="records", lines=True)
    queries.to_json(query_path, orient="records", lines=True)
