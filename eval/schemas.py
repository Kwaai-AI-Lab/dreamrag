from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievalCase:
    """Single benchmark query and its expected relevant IDs."""

    # Stable ID for the benchmark query.
    # This lets results and labels be matched reliably.
    query_id: str

    # The natural-language query being evaluated.
    query: str

    # Set of IDs that are considered relevant for this query.
    # These may represent chunks, documents, graph nodes, or other retrievable units.
    relevant_ids: set[str]


@dataclass(frozen=True)
class RetrievalResult:
    """Retrieved IDs for a benchmark query, ordered by rank."""

    # ID of the query this result belongs to.
    query_id: str

    # Retrieved item IDs in ranked order.
    # Index 0 is the top-ranked retrieval result.
    retrieved_ids: list[str]


@dataclass(frozen=True)
class GraphSnapshot:
    """Small graph summary captured before or after refinement."""

    # Number of nodes in the graph at this point in time.
    node_count: int

    # Number of edges/relationships in the graph at this point in time.
    edge_count: int

    # Number of duplicate entities or relations detected.
    duplicate_count: int = 0

    # Number of expected entities currently represented in the graph.
    covered_entities: int | None = None

    # Total number of entities expected for the benchmark or dataset.
    expected_entities: int | None = None

    # Number of valid relations currently represented in the graph.
    valid_relations: int | None = None

    # Total number of relations expected for the benchmark or dataset.
    expected_relations: int | None = None

    # Extra information that may be useful later, such as:
    # dream cycle number, dataset version, timestamp, or experiment name.
    metadata: dict[str, Any] = field(default_factory=dict)