"""
improved_graph_retrieval.py — Enhanced entity-aware retrieval.

Improvements:
1. Better query entity extraction
2. Confidence-weighted scoring
3. Relation type weighting
4. Deeper graph traversal
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from improved_extraction import ImprovedEntityExtractor, ExtractedEntity
from graph import GraphStore
from gliner_ner import DEFAULT_THRESHOLD, DEFAULT_URL

logger = logging.getLogger(__name__)


class ImprovedGraphRetriever:
    """Enhanced entity-aware retrieval with better matching and scoring."""

    def __init__(
        self,
        graph_store: GraphStore,
        chunk_to_doc: Optional[Dict[int, str]] = None,
        gliner_url: Optional[str] = DEFAULT_URL,
    ):
        """Initialize improved graph retriever."""
        self.graph = graph_store
        self.chunk_to_doc = chunk_to_doc or {}
        self.extractor = ImprovedEntityExtractor(
            gliner_url=gliner_url,
            gliner_threshold=DEFAULT_THRESHOLD,
            use_pattern_fallback=False,
        )

        # Relation type weights (semantic relations worth more)
        self.relation_weights = {
            # High-value semantic relations
            "spouse_of": 1.0,
            "parent_of": 1.0,
            "child_of": 1.0,
            "sibling_of": 0.9,
            "founded": 1.0,
            "led": 0.9,
            "worked_at": 0.8,
            "studied_under": 0.9,
            "collaborated_with": 0.7,
            "discovered": 1.0,
            "invented": 1.0,
            "capital_of": 0.9,
            "located_in": 0.8,
            "part_of": 0.8,
            # Low-value relations
            "co_occurs": 0.3,  # Much lower for co-occurrence
        }

    def search(
        self,
        query: str,
        top_k: int = 10,
        max_depth: int = 2,
        min_confidence: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Search using improved graph traversal.

        Args:
            query: Query text
            top_k: Number of results
            max_depth: Maximum traversal depth
            min_confidence: Minimum entity confidence threshold

        Returns:
            List of (doc_id, score) tuples
        """
        # Extract entities from query with confidence threshold
        query_entities = self._extract_query_entities(query, min_confidence)

        if not query_entities:
            logger.warning("No entities extracted from query: %s", query)
            return []

        logger.info("Found %d high-confidence entities in query", len(query_entities))

        # Traverse graph to find relevant documents
        doc_scores: Dict[str, float] = defaultdict(float)
        doc_metadata: Dict[str, tuple] = {}

        # Phase 1: Direct document retrieval
        direct_results = self._find_direct_documents(query_entities)
        for doc_id, score in direct_results:
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = score
                doc_metadata[doc_id] = (0, [e.name for e in query_entities])

        # Phase 2: Graph traversal
        if max_depth > 0:
            traversal_results = self._traverse_graph(
                query_entities, max_depth, min_confidence
            )
            for doc_id, (score, depth, entities) in traversal_results.items():
                if doc_id not in doc_scores or score > doc_scores[doc_id]:
                    doc_scores[doc_id] = score
                    doc_metadata[doc_id] = (depth, entities)

        # Compute final scores with depth penalty
        final_scores: List[Tuple[str, float]] = []
        for doc_id, raw_score in doc_scores.items():
            depth = doc_metadata[doc_id][0]
            # Penalize by depth (but don't zero out far results)
            final_score = raw_score / (1.0 + 0.3 * depth)
            final_scores.append((doc_id, final_score))

        # Sort and return top-k
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:top_k]

    def _extract_query_entities(
        self, query: str, min_confidence: float = 0.5
    ) -> List[ExtractedEntity]:
        """Extract high-confidence entities from query."""
        entities = self.extractor.extract_entities(query)

        # Filter by confidence and find matches in graph
        matched_entities = []
        for entity in entities:
            if entity.confidence >= min_confidence:
                # Try to find exact match in graph
                graph_node = self.graph.find_by_name(entity.name)
                if graph_node:
                    # Update with graph confidence
                    entity.confidence = max(entity.confidence, graph_node.confidence)
                    matched_entities.append(entity)
                elif entity.confidence >= 0.75:
                    # Keep very high confidence entities even if not in graph
                    matched_entities.append(entity)

        return matched_entities

    def _find_direct_documents(
        self, entities: List[ExtractedEntity]
    ) -> List[Tuple[str, float]]:
        """Find documents directly mentioning query entities."""
        doc_scores: Dict[str, float] = defaultdict(float)

        for entity in entities:
            graph_node = self.graph.find_by_name(entity.name)
            if not graph_node:
                continue

            # Get chunks for this entity
            chunk_ids = self.graph._entity_to_chunks.get(graph_node.id, [])
            for chunk_id in chunk_ids:
                doc_id = self.chunk_to_doc.get(chunk_id)
                if doc_id:
                    # Score based on entity confidence
                    score = entity.confidence * graph_node.confidence
                    doc_scores[doc_id] = max(doc_scores[doc_id], score)

        return list(doc_scores.items())

    def _traverse_graph(
        self,
        query_entities: List[ExtractedEntity],
        max_depth: int,
        min_confidence: float,
    ) -> Dict[str, Tuple[float, int, List[str]]]:
        """Traverse graph to find documents via entity neighbors."""
        neighbor_docs: Dict[str, Tuple[float, int, List[str]]] = {}

        # Convert to set of entity IDs
        visited: Set[int] = set()
        current_frontier: List[Tuple[int, float, str]] = []  # (entity_id, confidence, name)

        for entity in query_entities:
            node = self.graph.find_by_name(entity.name)
            if node:
                visited.add(node.id)
                current_frontier.append((node.id, entity.confidence, entity.name))

        for depth in range(1, max_depth + 1):
            next_frontier = []

            for entity_id, entity_conf, entity_name in current_frontier:
                # Get neighbors
                relations = self.graph.get_relations(entity_id)
                for neighbor_id, relation_type, relation_strength in relations:
                    if neighbor_id in visited:
                        continue

                    neighbor_node = self.graph.get_entity(neighbor_id)
                    if not neighbor_node or neighbor_node.confidence < min_confidence:
                        continue

                    visited.add(neighbor_id)

                    # Score based on relation weight
                    rel_weight = self.relation_weights.get(relation_type, 0.5)
                    score = (
                        entity_conf
                        * neighbor_node.confidence
                        * relation_strength
                        * rel_weight
                    )

                    # Decay by depth
                    decay = 1.0 / (depth + 1)
                    final_score = score * decay

                    if final_score < 0.05:  # Skip very low scores
                        continue

                    # Find documents for this neighbor
                    chunk_ids = self.graph._entity_to_chunks.get(neighbor_id, [])
                    for chunk_id in chunk_ids:
                        doc_id = self.chunk_to_doc.get(chunk_id)
                        if doc_id:
                            if doc_id not in neighbor_docs or final_score > neighbor_docs[doc_id][0]:
                                neighbor_docs[doc_id] = (final_score, depth, [neighbor_node.name])

                    # Add to next frontier
                    next_frontier.append((neighbor_id, final_score, neighbor_node.name))

            if not next_frontier:
                break

            current_frontier = next_frontier

        return neighbor_docs

    def stats(self) -> Dict:
        """Return retriever statistics."""
        return {
            "entities": self.graph.node_count(),
            "relations": self.graph.relation_count(),
            "documents": len(set(self.chunk_to_doc.values())),
            "extractor_has_spacy": self.extractor.spacy_available,
        }
