"""
graph_retrieval.py — Entity-aware retrieval using the knowledge graph.

Implements graph-based document ranking by:
1. Extracting entities from query
2. Finding matching nodes in graph
3. Traversing neighbors for cross-document retrieval
4. Scoring by entity strength + document relevance
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

from graph import GraphStore, EntityNode, normalize_name

logger = logging.getLogger(__name__)


@dataclass
class EntityMatch:
    """Represents a matched entity from query."""
    entity_id: int
    entity_name: str
    entity_type: str
    confidence: float
    chunk_ids: List[int]


@dataclass
class GraphRetrievalResult:
    """Result from graph-based retrieval."""
    doc_id: str
    score: float
    matched_entities: List[int]  # Entity IDs found in this document
    relation_depth: int  # How many hops from query entities
    direct_mention: bool  # Directly mentions query entity


class GraphRetriever:
    """Entity-aware document retriever using knowledge graph."""

    def __init__(self, graph_store: GraphStore, chunk_to_doc: Optional[Dict[int, str]] = None):
        """
        Initialize graph retriever.

        Args:
            graph_store: GraphStore instance with populated entities and relations
            chunk_to_doc: Mapping from chunk_id to document_id (required for retrieval)
        """
        self.graph = graph_store
        self.chunk_to_doc = chunk_to_doc or {}

    def search(
        self,
        query: str,
        top_k: int = 10,
        entity_weight: float = 0.5,
        proximity_weight: float = 0.5,
        max_depth: int = 2,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve documents using entity-aware graph traversal.

        Args:
            query: Query text
            top_k: Number of documents to return
            entity_weight: Weight for entity match score (0-1)
            proximity_weight: Weight for proximity/depth score (0-1)
            max_depth: Maximum graph traversal depth (hops from query entities)

        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        # Extract entities from query
        query_entities = self._extract_entities_from_query(query)
        if not query_entities:
            logger.warning("No entities extracted from query: %s", query)
            return []

        logger.info("Found %d entities in query", len(query_entities))

        # Traverse graph to find related documents
        doc_scores: Dict[str, float] = defaultdict(float)
        doc_metadata: Dict[str, GraphRetrievalResult] = {}

        for matched_entity in query_entities:
            # Find documents directly mentioning this entity
            direct_docs = self._find_documents_for_entity(matched_entity.entity_id)
            for doc_id, chunk_ids in direct_docs.items():
                entity_score = matched_entity.confidence
                current_score = doc_scores[doc_id]
                doc_scores[doc_id] = max(current_score, entity_score)

                if doc_id not in doc_metadata:
                    doc_metadata[doc_id] = GraphRetrievalResult(
                        doc_id=doc_id,
                        score=0.0,
                        matched_entities=[],
                        relation_depth=0,
                        direct_mention=True,
                    )
                doc_metadata[doc_id].matched_entities.append(matched_entity.entity_id)

        # Traverse to neighboring entities (cross-document retrieval)
        neighbor_docs = self._traverse_neighbors(
            query_entities,
            max_depth=max_depth,
            visited=set(q.entity_id for q in query_entities),
        )

        for doc_id, (neighbor_score, depth, neighbor_entities) in neighbor_docs.items():
            if doc_id not in doc_scores:
                doc_scores[doc_id] = neighbor_score

                doc_metadata[doc_id] = GraphRetrievalResult(
                    doc_id=doc_id,
                    score=0.0,
                    matched_entities=neighbor_entities,
                    relation_depth=depth,
                    direct_mention=False,
                )

        # Compute final scores using weights
        final_scores: List[Tuple[str, float]] = []
        for doc_id, score in doc_scores.items():
            metadata = doc_metadata[doc_id]

            if metadata.direct_mention:
                final_score = entity_weight * score + proximity_weight * 1.0
            else:
                proximity_bonus = 1.0 / (metadata.relation_depth + 1)
                final_score = entity_weight * 0.5 + proximity_weight * proximity_bonus

            final_scores.append((doc_id, final_score))

        # Sort by score and return top-k
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:top_k]

    def _extract_entities_from_query(self, query: str) -> List[EntityMatch]:
        """Extract entity matches from query text using graph lookups."""
        query_tokens = query.lower().split()
        matched_entities: Dict[int, EntityMatch] = {}

        # Try to match tokens/phrases to entities in graph
        for token in query_tokens:
            # Exact name match
            entity = self.graph.find_by_name(token)
            if entity and entity.id not in matched_entities:
                chunk_ids = self.graph._entity_to_chunks.get(entity.id, [])
                matched_entities[entity.id] = EntityMatch(
                    entity_id=entity.id,
                    entity_name=entity.name,
                    entity_type=entity.entity_type,
                    confidence=entity.confidence,
                    chunk_ids=chunk_ids,
                )

        # Also try multi-word phrases (simple 2-3 word combinations)
        tokens_lower = query.lower().split()
        for i in range(len(tokens_lower) - 1):
            phrase = " ".join(tokens_lower[i : i + 2])
            entity = self.graph.find_by_name(phrase)
            if entity and entity.id not in matched_entities:
                chunk_ids = self.graph._entity_to_chunks.get(entity.id, [])
                matched_entities[entity.id] = EntityMatch(
                    entity_id=entity.id,
                    entity_name=entity.name,
                    entity_type=entity.entity_type,
                    confidence=entity.confidence,
                    chunk_ids=chunk_ids,
                )

        return list(matched_entities.values())

    def _find_documents_for_entity(self, entity_id: int) -> Dict[str, List[int]]:
        """Find all documents containing this entity."""
        chunk_ids = self.graph._entity_to_chunks.get(entity_id, [])
        docs: Dict[str, List[int]] = defaultdict(list)

        for chunk_id in chunk_ids:
            doc_id = self.chunk_to_doc.get(chunk_id)
            if doc_id:
                docs[doc_id].append(chunk_id)

        return docs

    def _traverse_neighbors(
        self,
        query_entities: List[EntityMatch],
        max_depth: int = 2,
        visited: Optional[Set[int]] = None,
    ) -> Dict[str, Tuple[float, int, List[int]]]:
        """
        Traverse graph to find neighboring entities and their documents.

        Returns:
            Dict[doc_id] = (score, depth, entity_ids)
        """
        visited = visited or set()
        neighbor_docs: Dict[str, Tuple[float, int, List[int]]] = {}

        for depth in range(1, max_depth + 1):
            new_neighbors: Set[int] = set()

            for entity in query_entities:
                # Get neighbors of this entity
                relations = self.graph.get_relations(entity.entity_id)

                for neighbor_id, relation_type, strength in relations:
                    if neighbor_id in visited:
                        continue

                    neighbor_node = self.graph.get_entity(neighbor_id)
                    if not neighbor_node:
                        continue

                    new_neighbors.add(neighbor_id)
                    visited.add(neighbor_id)

                    # Find documents for this neighbor
                    neighbor_docs_for_entity = self._find_documents_for_entity(neighbor_id)

                    for doc_id, chunk_ids in neighbor_docs_for_entity.items():
                        # Score based on relation strength and neighbor confidence
                        score = strength * neighbor_node.confidence

                        # Decay score by depth
                        decay_factor = 1.0 / (depth + 1)
                        final_score = score * decay_factor

                        if doc_id not in neighbor_docs:
                            neighbor_docs[doc_id] = (final_score, depth, [neighbor_id])
                        else:
                            existing_score, existing_depth, existing_entities = neighbor_docs[doc_id]
                            if final_score > existing_score:
                                neighbor_docs[doc_id] = (
                                    final_score,
                                    depth,
                                    existing_entities + [neighbor_id],
                                )

            # Update query_entities for next iteration
            if new_neighbors:
                for eid in new_neighbors:
                    node = self.graph.get_entity(eid)
                    if node:
                        chunk_ids = self.graph._entity_to_chunks.get(eid, [])
                        query_entities.append(
                            EntityMatch(
                                entity_id=eid,
                                entity_name=node.name,
                                entity_type=node.entity_type,
                                confidence=node.confidence,
                                chunk_ids=chunk_ids,
                            )
                        )

        return neighbor_docs

    def stats(self) -> Dict:
        """Return graph statistics."""
        return {
            "total_entities": self.graph.node_count(),
            "total_relations": self.graph.relation_count(),
            "documents_mapped": len(set(self.chunk_to_doc.values())),
        }
