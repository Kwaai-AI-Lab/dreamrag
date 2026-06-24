from __future__ import annotations
from .schemas import GraphSnapshot


def entity_coverage(covered_entities: int, expected_entities: int) -> float:
    """Compute entity coverage as covered_entities / expected_entities."""

    # expected_entities is the total number of entities we believe should exist.
    # If this is zero or invalid, we avoid division by zero.
    if expected_entities <= 0:
        return 0.0

    # A higher value means the graph contains more of the expected knowledge.
    return covered_entities / expected_entities


def relation_completeness(valid_relations: int, expected_relations: int) -> float:
    """Compute relation completeness as valid_relations / expected_relations."""

    # expected_relations is the total number of relationships we expect to find.
    # If this is zero or invalid, we avoid division by zero.
    if expected_relations <= 0:
        return 0.0

    # A higher value means the graph captures more expected relationships.
    return valid_relations / expected_relations


def compression_delta(before: GraphSnapshot, after: GraphSnapshot) -> dict[str, float]:
    """Measure graph size change after refinement."""

    # Compare node counts before and after refinement.
    # Positive values mean the graph became smaller.
    node_reduction = _relative_reduction(before.node_count, after.node_count)

    # Compare edge counts before and after refinement.
    # Positive values mean relationships were compressed or pruned.
    edge_reduction = _relative_reduction(before.edge_count, after.edge_count)

    # Return both raw counts and reduction ratios.
    # Raw counts help debugging; ratios help compare different graph sizes.
    return {
        "nodes_before": float(before.node_count),
        "nodes_after": float(after.node_count),
        "node_reduction_ratio": node_reduction,
        "edges_before": float(before.edge_count),
        "edges_after": float(after.edge_count),
        "edge_reduction_ratio": edge_reduction,
    }


def deduplication_delta(before: GraphSnapshot, after: GraphSnapshot) -> dict[str, float]:
    """Measure duplicate reduction after refinement."""

    # Compare duplicate counts before and after refinement.
    # Positive values mean duplicate entities/relations were reduced.
    duplicate_reduction = _relative_reduction(
        before.duplicate_count,
        after.duplicate_count,
    )

    # Return duplicate counts and the relative reduction.
    return {
        "duplicates_before": float(before.duplicate_count),
        "duplicates_after": float(after.duplicate_count),
        "duplicate_reduction_ratio": duplicate_reduction,
    }


def snapshot_entity_coverage(snapshot: GraphSnapshot) -> float:
    """Compute entity coverage from a GraphSnapshot when available."""

    # Some snapshots may not include entity coverage fields.
    # If either value is missing, return 0 instead of failing.
    if snapshot.covered_entities is None or snapshot.expected_entities is None:
        return 0.0

    # Reuse the main entity_coverage function for consistent behavior.
    return entity_coverage(snapshot.covered_entities, snapshot.expected_entities)


def snapshot_relation_completeness(snapshot: GraphSnapshot) -> float:
    """Compute relation completeness from a GraphSnapshot when available."""

    # Some snapshots may not include relation completeness fields.
    # If either value is missing, return 0 instead of failing.
    if snapshot.valid_relations is None or snapshot.expected_relations is None:
        return 0.0

    # Reuse the main relation_completeness function for consistent behavior.
    return relation_completeness(snapshot.valid_relations, snapshot.expected_relations)


def _relative_reduction(before: int, after: int) -> float:
    # This helper calculates how much a value decreased.
    # Example: before=100 and after=75 gives 0.25, meaning 25% reduction.

    # If before is zero or invalid, reduction cannot be calculated safely.
    if before <= 0:
        return 0.0

    # Positive value: reduction.
    # Zero: no change.
    # Negative value: expansion.
    return (before - after) / before