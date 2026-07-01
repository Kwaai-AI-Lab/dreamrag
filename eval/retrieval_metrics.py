from __future__ import annotations

import math
from collections.abc import Mapping

from .schemas import RetrievalCase, RetrievalResult


def _case_map(cases: list[RetrievalCase]) -> dict[str, RetrievalCase]:
    """Map benchmark cases by query_id for lookup."""
    # Convert the list of benchmark cases into a dictionary.
    # This lets us quickly look up the correct answer set for a query_id.
    
    return {case.query_id: case for case in cases}


def recall_at_k(
    cases: list[RetrievalCase],
    results: list[RetrievalResult],
    k: int,
) -> float:
    """
    Compute query-level Recall@K.
    A query is counted as successful if at least one relevant item appears within the first K retrieved results.
    """
    # K represents how many retrieved results we inspect.
    # For example, K=5 means "look at the top 5 retrieved items."
    if k <= 0:
        raise ValueError("k must be positive")

    # If there are no benchmark questions, there is nothing to score.
    if not cases:
        return 0.0

    # Build a lookup table so each result can be matched to its benchmark case.
    cases_by_id = _case_map(cases)

    # Count how many queries successfully retrieved at least one relevant item.
    hits = 0

    for result in results:
        # Find the benchmark case that corresponds to this retrieval result.
        case = cases_by_id.get(result.query_id)

        # Skip results that do not match any known benchmark case.
        if case is None:
            continue

        # Keep only the top K retrieved IDs.
        # These are the results the user/model would most likely see first.
        top_k = set(result.retrieved_ids[:k])

        # If any retrieved item is relevant, this query counts as a hit.
        if top_k.intersection(case.relevant_ids):
            hits += 1

    # Recall@K is the fraction of benchmark queries that had a hit.
    return hits / len(cases)


def mean_reciprocal_rank(
    cases: list[RetrievalCase],
    results: list[RetrievalResult],
) -> float:
    """
    Compute Mean Reciprocal Rank across benchmark queries.
    MRR rewards systems that place the first relevant result higher in the ranked retrieval list.
    """
    # If there are no benchmark questions, there is nothing to score.
    if not cases:
        return 0.0

    # Build a lookup table so each result can be matched to its benchmark case.
    cases_by_id = _case_map(cases)

    # Store one reciprocal-rank score per query.
    reciprocal_ranks: list[float] = []

    for result in results:
        # Find the benchmark case for this retrieval result.
        case = cases_by_id.get(result.query_id)

        # Ignore retrieval results that do not correspond to a known query.
        if case is None:
            continue

        # Default score is 0 if no relevant item appears in the retrieved list.
        rank_score = 0.0

        # Walk through retrieved IDs in ranked order.
        # enumerate(..., start=1) gives human-readable ranks: 1, 2, 3, ...
        for index, retrieved_id in enumerate(result.retrieved_ids, start=1):

            # The first relevant item determines the reciprocal-rank score.
            if retrieved_id in case.relevant_ids:
                # Rank 1 gives 1.0, rank 2 gives 0.5, rank 10 gives 0.1.
                rank_score = 1.0 / index
                break

        # Save this query's reciprocal-rank score.
        reciprocal_ranks.append(rank_score)

    # If some benchmark cases had no retrieval result, count them as 0.
    missing = len(cases) - len(reciprocal_ranks)
    reciprocal_ranks.extend([0.0] * max(missing, 0))

    # MRR is the average reciprocal-rank score across all benchmark queries.
    return sum(reciprocal_ranks) / len(cases)


def ndcg_at_k(
    relevance_by_query: Mapping[str, Mapping[str, float]],
    results: list[RetrievalResult],
    k: int,
) -> float:
    """
    Compute average nDCG@K.
    nDCG evaluates ranked retrieval quality using binary or graded relevance labels.
    """
    # K represents how many ranked results we evaluate.
    if k <= 0:
        raise ValueError("k must be positive")

    # If no relevance labels are provided, nDCG cannot be calculated.
    if not relevance_by_query:
        return 0.0

    # Store one nDCG score per query.
    ndcg_scores: list[float] = []

    for result in results:
        # Get the relevance labels for this query.
        # Example: {"doc1": 3.0, "doc2": 1.0}
        relevance_map = relevance_by_query.get(result.query_id, {})

        # If the query has no relevance labels, score it as 0.
        if not relevance_map:
            ndcg_scores.append(0.0)
            continue

        # Convert the retrieved IDs into relevance scores.
        # Unlabeled retrieved items receive a relevance score of 0.
        retrieved_relevances = [
            relevance_map.get(item_id, 0.0)
            for item_id in result.retrieved_ids[:k]
        ]

        # DCG measures how useful the actual ranked list is.
        dcg = _dcg(retrieved_relevances)

        # The ideal ranking puts the most relevant items first.
        ideal_relevances = sorted(relevance_map.values(), reverse=True)[:k]

        # IDCG measures the best possible ranked list for this query.
        idcg = _dcg(ideal_relevances)

        # nDCG compares actual ranking quality against the ideal ranking.
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    # Return the average nDCG score across all evaluated queries.
    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


def _dcg(relevances: list[float]) -> float:
    """Compute Discounted Cumulative Gain for a ranked relevance list."""
    # DCG stands for Discounted Cumulative Gain.
    # It rewards relevant results, especially when they appear near the top.
    score = 0.0

    for index, relevance in enumerate(relevances, start=1):
        # Higher relevance increases the score.
        # The log denominator discounts results that appear lower in the ranking.
        score += (2**relevance - 1) / math.log2(index + 1)

    return score