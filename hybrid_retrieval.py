#!/usr/bin/env python3
"""
hybrid_retrieval.py — Combine BM25 lexical + graph-based entity retrieval.

Uses reciprocal rank fusion to merge results from:
1. BM25 lexical matching
2. Graph-based entity retrieval
3. Document proximity in entity network
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from eval.retrieval_rankers import BM25Retriever, reciprocal_rank_fusion
from eval.schemas import RetrievalResult
from graph import GraphStore
from graph_retrieval import GraphRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining lexical and graph-based methods."""

    def __init__(
        self,
        documents: Dict[str, str],
        graph_db_path: Optional[str] = None,
        chunk_mapping_path: Optional[str] = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            documents: Dict[doc_id, text]
            graph_db_path: Path to graph database
            chunk_mapping_path: Path to chunk-to-doc mapping JSON
        """
        self.documents = documents
        self.bm25 = BM25Retriever(documents)

        self.graph_retriever: Optional[GraphRetriever] = None
        if graph_db_path and Path(graph_db_path).exists():
            try:
                graph = GraphStore(graph_db_path)
                chunk_to_doc = {}
                if chunk_mapping_path and Path(chunk_mapping_path).exists():
                    with open(chunk_mapping_path) as f:
                        chunk_to_doc = {int(k): v for k, v in json.load(f).items()}

                self.graph_retriever = GraphRetriever(graph, chunk_to_doc)
                logger.info("Graph retriever initialized")
            except Exception as e:
                logger.warning("Failed to initialize graph retriever: %s", e)

    def search(
        self,
        query_id: str,
        query: str,
        top_k: int = 10,
        bm25_weight: float = 0.6,
        graph_weight: float = 0.4,
    ) -> RetrievalResult:
        """
        Search using hybrid approach.

        Args:
            query_id: Unique query identifier
            query: Query text
            top_k: Number of results to return
            bm25_weight: Weight for BM25 results (0-1)
            graph_weight: Weight for graph results (0-1)

        Returns:
            RetrievalResult with merged ranking
        """
        results: List[RetrievalResult] = []

        # BM25 search
        try:
            bm25_result = self.bm25.search(query_id, query, top_k=top_k)
            results.append(bm25_result)
            logger.debug("BM25 returned %d results", len(bm25_result.retrieved_ids))
        except Exception as e:
            logger.warning("BM25 search failed: %s", e)

        # Graph-based search
        if self.graph_retriever:
            try:
                graph_docs = self.graph_retriever.search(
                    query=query,
                    top_k=top_k,
                    entity_weight=0.6,
                    proximity_weight=0.4,
                )
                graph_result = RetrievalResult(
                    query_id=query_id,
                    retrieved_ids=[doc_id for doc_id, _ in graph_docs],
                )
                results.append(graph_result)
                logger.debug("Graph returned %d results", len(graph_result.retrieved_ids))
            except Exception as e:
                logger.warning("Graph search failed: %s", e)

        # Merge results using RRF
        if len(results) > 1:
            merged = reciprocal_rank_fusion(
                query_id=query_id,
                ranked_results=results,
                top_k=top_k,
            )
            return merged
        elif results:
            return results[0]
        else:
            return RetrievalResult(query_id=query_id, retrieved_ids=[])

    def stats(self) -> Dict:
        """Return retriever statistics."""
        stats = {
            "documents": len(self.documents),
            "bm25_initialized": self.bm25 is not None,
            "graph_initialized": self.graph_retriever is not None,
        }
        if self.graph_retriever:
            stats.update(self.graph_retriever.stats())
        return stats
