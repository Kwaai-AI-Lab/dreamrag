"""
Optional RAGAS runner for Dream RAG retrieval-grounding evaluation.

Requires optional dependencies:
    pip install ragas datasets
"""
from __future__ import annotations
from .ragas_adapter import RagasCase, ragas_metric_names, to_ragas_dataset


def run_ragas_evaluation(cases: list[RagasCase]):
    """
    Run RAGAS retrieval-grounding metrics.

    Evaluates:
        - Context Recall
        - Context Precision
        - Faithfulness
    """

    # Import here so RAGAS is optional for the rest of Dream RAG.
    try:
        from ragas import evaluate
        from ragas.metrics import context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise ImportError(
            "RAGAS evaluation requires `ragas` and `datasets`. "
            "Install with: pip install ragas datasets"
        ) from exc

    # Convert Dream RAG evaluation records into RAGAS format.
    dataset = to_ragas_dataset(cases)

    # Run only the metrics currently scoped for retrieval grounding.
    return evaluate(
        dataset,
        metrics=[
            context_recall,
            context_precision,
            faithfulness,
        ],
    )


def available_ragas_metrics() -> list[str]:
    """Return the RAGAS metrics exposed by this runner."""

    return ragas_metric_names()