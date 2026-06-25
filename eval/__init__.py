"""
Dream RAG Evaluation Package

This package contains evaluation utilities used to assess:

1. Retrieval Effectiveness
2. Retrieval Grounding
3. Graph Refinement Quality

Current Retrieval Effectiveness Metrics:
    - Recall@K
    - Mean Reciprocal Rank (MRR)
    - Normalized Discounted Cumulative Gain (nDCG)

Current Retrieval Grounding Metrics:
    - Context Recall
    - Context Precision
    - Faithfulness

Current Graph Refinement Metrics:
    - Entity Coverage
    - Relation Completeness
    - Compression Delta
    - Deduplication Delta

Future evaluation modules may include:

    - Trustworthiness Evaluation
    - Security Evaluation
    - Alignment & Safety Evaluation

These remain intentionally outside the current retrieval-quality
and retrieval-grounding scope.
"""

# ---------------------------------------------------------------------
# Evaluation Data Structures
# ---------------------------------------------------------------------

from .schemas import (
    RetrievalCase,
    RetrievalResult,
    GraphSnapshot,
)

# ---------------------------------------------------------------------
# Retrieval Effectiveness Metrics
# ---------------------------------------------------------------------

from .retrieval_metrics import (
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)

# ---------------------------------------------------------------------
# Graph Refinement Metrics
# ---------------------------------------------------------------------

from .graph_metrics import (
    entity_coverage,
    relation_completeness,
    compression_delta,
    deduplication_delta,
)

# ---------------------------------------------------------------------
# Retrieval Grounding (RAGAS)
# ---------------------------------------------------------------------

from .ragas_adapter import (
    RagasCase,
    to_ragas_dataset,
    ragas_metric_names,
)

from .ragas_runner import (
    run_ragas_evaluation,
    available_ragas_metrics,
)

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

__all__ = [
    # Schemas
    "RetrievalCase",
    "RetrievalResult",
    "GraphSnapshot",

    # Retrieval Effectiveness
    "recall_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",

    # Graph Refinement
    "entity_coverage",
    "relation_completeness",
    "compression_delta",
    "deduplication_delta",

    # Retrieval Grounding
    "RagasCase",
    "to_ragas_dataset",
    "ragas_metric_names",
    "run_ragas_evaluation",
    "available_ragas_metrics",
]