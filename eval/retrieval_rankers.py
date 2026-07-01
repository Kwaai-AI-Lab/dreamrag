from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .schemas import RetrievalResult


@dataclass(frozen=True)
class RankedDocument:
    doc_id: str
    text: str
    score: float = 0.0
    metadata: dict | None = None


class BM25Retriever:
    """
    Lightweight BM25 lexical retriever for DreamRAG evaluation.

    Use this to produce RetrievalResult objects that can be scored by:
    - recall_at_k
    - mean_reciprocal_rank
    - ndcg_at_k
    """

    def __init__(
        self,
        documents: Mapping[str, str],
        tokenizer: Callable[[str], list[str]] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.documents = dict(documents)
        self.tokenizer = tokenizer or self._default_tokenizer
        self.k1 = k1
        self.b = b

        self.doc_tokens = {
            doc_id: self.tokenizer(text)
            for doc_id, text in self.documents.items()
        }

        self.term_freqs: dict[str, Counter] = {}
        self.doc_freqs: dict[str, int] = defaultdict(int)

        for doc_id, tokens in self.doc_tokens.items():
            counts = Counter(tokens)
            self.term_freqs[doc_id] = counts

            for term in counts:
                self.doc_freqs[term] += 1

        self.num_docs = len(self.documents)
        self.avg_doc_len = (
            sum(len(tokens) for tokens in self.doc_tokens.values()) / self.num_docs
            if self.num_docs
            else 0.0
        )

    @staticmethod
    def _default_tokenizer(text: str) -> list[str]:
        return text.lower().split()

    def idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        return math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))

    def score_document(self, query: str, doc_id: str) -> float:
        if doc_id not in self.documents:
            raise KeyError(f"Unknown doc_id: {doc_id}")

        query_terms = self.tokenizer(query)
        doc_len = len(self.doc_tokens[doc_id])
        freqs = self.term_freqs[doc_id]

        if self.avg_doc_len == 0:
            return 0.0

        score = 0.0

        for term in query_terms:
            tf = freqs.get(term, 0)

            if tf == 0:
                continue

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_len / self.avg_doc_len
            )

            score += self.idf(term) * numerator / denominator

        return score

    def search(
        self,
        query_id: str,
        query: str,
        top_k: int = 10,
    ) -> RetrievalResult:
        scored = [
            RankedDocument(
                doc_id=doc_id,
                text=text,
                score=self.score_document(query, doc_id),
            )
            for doc_id, text in self.documents.items()
        ]

        ranked = sorted(scored, key=lambda item: item.score, reverse=True)

        return RetrievalResult(
            query_id=query_id,
            retrieved_ids=[item.doc_id for item in ranked[:top_k]],
        )


def reciprocal_rank_fusion(
    query_id: str,
    ranked_results: Sequence[RetrievalResult],
    k: int = 60,
    top_k: int = 10,
) -> RetrievalResult:
    """
    Fuse multiple ranked RetrievalResult objects for the same query.

    Useful for combining:
    - BM25 lexical results
    - vector retrieval results
    - graph retrieval results
    - LTR reranked results
    """

    scores: dict[str, float] = defaultdict(float)

    for result in ranked_results:
        if result.query_id != query_id:
            continue

        for rank, doc_id in enumerate(result.retrieved_ids, start=1):
            scores[doc_id] += 1.0 / (k + rank)

    fused_ids = [
        doc_id
        for doc_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]

    return RetrievalResult(
        query_id=query_id,
        retrieved_ids=fused_ids[:top_k],
    )


class LinearLTRRanker:
    """
    Simple learning-to-rank baseline.

    This is intentionally lightweight. It supports feature-weighted ranking
    without adding sklearn/xgboost/lightgbm dependencies.

    Example features:
    - bm25_score
    - vector_score
    - graph_distance_score
    - entity_overlap_score
    - recency_score
    - provenance_score
    """

    def __init__(self, feature_weights: Mapping[str, float]):
        self.feature_weights = dict(feature_weights)

    def score_features(self, features: Mapping[str, float]) -> float:
        return sum(
            features.get(name, 0.0) * weight
            for name, weight in self.feature_weights.items()
        )

    def rerank(
        self,
        query_id: str,
        candidate_ids: Sequence[str],
        feature_lookup: Mapping[str, Mapping[str, float]],
        top_k: int = 10,
    ) -> RetrievalResult:
        scored = []

        for doc_id in candidate_ids:
            features = feature_lookup.get(doc_id, {})
            score = self.score_features(features)
            scored.append((doc_id, score))

        ranked_ids = [
            doc_id
            for doc_id, _ in sorted(scored, key=lambda item: item[1], reverse=True)
        ]

        return RetrievalResult(
            query_id=query_id,
            retrieved_ids=ranked_ids[:top_k],
        )


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def maximal_marginal_relevance(
    query_id: str,
    query_embedding: Sequence[float],
    candidate_ids: Sequence[str],
    document_embeddings: Mapping[str, Sequence[float]],
    lambda_mult: float = 0.5,
    top_k: int = 10,
) -> RetrievalResult:
    """
    Diversify retrieved results while preserving query relevance.

    lambda_mult closer to 1.0 favors relevance.
    lambda_mult closer to 0.0 favors diversity.
    """

    selected: list[str] = []
    remaining = list(candidate_ids)

    while remaining and len(selected) < top_k:
        best_id = None
        best_score = float("-inf")

        for doc_id in remaining:
            doc_embedding = document_embeddings[doc_id]

            relevance = cosine_similarity(query_embedding, doc_embedding)

            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    cosine_similarity(doc_embedding, document_embeddings[selected_id])
                    for selected_id in selected
                )

            mmr_score = (
                lambda_mult * relevance
                - (1 - lambda_mult) * diversity_penalty
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_id = doc_id

        selected.append(best_id)
        remaining.remove(best_id)

    return RetrievalResult(
        query_id=query_id,
        retrieved_ids=selected,
    )


class CrossEncoderReranker:
    """
    Cross-encoder reranker wrapper.

    Expected model interface:
        model.predict([(query, document_text), ...]) -> list[float]

    Compatible with sentence-transformers CrossEncoder.
    """

    def __init__(self, model):
        self.model = model

    def rerank(
        self,
        query_id: str,
        query: str,
        candidate_ids: Sequence[str],
        documents: Mapping[str, str],
        top_k: int = 10,
    ) -> RetrievalResult:
        pairs = [(query, documents[doc_id]) for doc_id in candidate_ids]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(candidate_ids, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        return RetrievalResult(
            query_id=query_id,
            retrieved_ids=[doc_id for doc_id, _ in ranked[:top_k]],
        )