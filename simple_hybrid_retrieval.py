"""
simple_hybrid_retrieval.py — Simple hybrid retrieval: BM25 + entity matching.

A conservative hybrid approach that boosts documents mentioning entities
extracted from the query (GLiNER / spaCy / patterns via ImprovedEntityExtractor).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from eval.retrieval_rankers import BM25Retriever
from eval.schemas import RetrievalResult
from improved_extraction import ImprovedEntityExtractor
from gliner_ner import DEFAULT_THRESHOLD, DEFAULT_URL

logger = logging.getLogger(__name__)


class SimpleHybridRetriever:
    """Simple hybrid combining BM25 with entity mention boosting."""

    def __init__(
        self,
        documents: Dict[str, str],
        gliner_url: Optional[str] = DEFAULT_URL,
    ):
        """Initialize hybrid retriever."""
        self.documents = documents
        self.bm25 = BM25Retriever(documents)
        self.extractor = ImprovedEntityExtractor(
            gliner_url=gliner_url,
            gliner_threshold=DEFAULT_THRESHOLD,
            use_pattern_fallback=False,
        )

        # Build entity mention index from queries (for demo)
        self.entity_mentions: Dict[str, List[str]] = defaultdict(list)

    def search(
        self,
        query_id: str,
        query: str,
        top_k: int = 10,
        bm25_weight: float = 0.7,
        entity_weight: float = 0.3,
    ) -> RetrievalResult:
        """
        Search with BM25 + entity mention boosting.

        Args:
            query_id: Query identifier
            query: Query text
            top_k: Number of results
            bm25_weight: Weight for BM25 scores
            entity_weight: Weight for entity match bonus

        Returns:
            RetrievalResult with merged ranking
        """
        # Get BM25 results
        bm25_result = self.bm25.search(query_id, query, top_k=top_k * 2)

        if not bm25_result.retrieved_ids:
            return RetrievalResult(query_id=query_id, retrieved_ids=[])

        # Prefer high-precision GLiNER entities; fall back to capitalized tokens
        extracted = self.extractor.extract_entities(query)
        entity_candidates = [e.name for e in extracted if e.confidence >= DEFAULT_THRESHOLD]
        if not entity_candidates:
            entity_candidates = [
                t for t in query.split()
                if t and t[0].isupper() and len(t) > 2
            ]

        # Score each BM25 result with entity bonus
        scored_results: List[Tuple[str, float]] = []
        for rank, doc_id in enumerate(bm25_result.retrieved_ids, 1):
            bm25_score = 1.0 / rank  # Simple rank-based score

            # Check how many entities appear in this document
            doc_text = self.documents.get(doc_id, "").lower()
            entity_matches = sum(
                1 for entity in entity_candidates
                if entity.lower() in doc_text
            )

            # Normalize entity bonus (0-1)
            entity_bonus = min(1.0, entity_matches / max(1, len(entity_candidates)))

            # Combine scores
            final_score = (
                bm25_weight * bm25_score +
                entity_weight * entity_bonus
            )

            scored_results.append((doc_id, final_score))

        # Sort by final score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return RetrievalResult(
            query_id=query_id,
            retrieved_ids=[doc_id for doc_id, _ in scored_results[:top_k]],
        )
