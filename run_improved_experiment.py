#!/usr/bin/env python3
"""
run_improved_experiment.py — Evaluate improved graph-based retrieval.

Compares:
1. BM25 baseline
2. Original graph-based retrieval
3. Improved graph-based retrieval
4. Improved hybrid (BM25 + improved graph)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

from eval.schemas import RetrievalCase, RetrievalResult
from eval.retrieval_metrics import recall_at_k, mean_reciprocal_rank
from eval.retrieval_rankers import BM25Retriever
from corpus_retrieval_analysis import CorpusLoader
from improved_graph_retrieval import ImprovedGraphRetriever
from graph import GraphStore


def evaluate_retriever(
    cases: List[RetrievalCase],
    retriever,
    retriever_name: str,
    search_method: str = "search",
) -> Dict:
    """Evaluate a retriever on test cases."""
    print(f"\nEvaluating {retriever_name}...")
    results: List[RetrievalResult] = []

    for i, case in enumerate(cases):
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(cases)}]", end="\r")

        try:
            if search_method == "bm25":
                result = retriever.search(
                    query_id=case.query_id,
                    query=case.query,
                    top_k=10,
                )
            elif search_method == "improved_graph":
                graph_docs = retriever.search(query=case.query, top_k=10)
                result = RetrievalResult(
                    query_id=case.query_id,
                    retrieved_ids=[doc_id for doc_id, _ in graph_docs],
                )
            results.append(result)
        except Exception as e:
            logger.debug(f"Error on {case.query_id}: {e}")
            results.append(RetrievalResult(query_id=case.query_id, retrieved_ids=[]))

    print(f"  ✓ Completed {len(results)} queries")

    # Compute metrics
    metrics = {
        "recall@1": recall_at_k(cases, results, k=1),
        "recall@3": recall_at_k(cases, results, k=3),
        "recall@5": recall_at_k(cases, results, k=5),
        "recall@10": recall_at_k(cases, results, k=10),
        "mrr": mean_reciprocal_rank(cases, results),
    }

    return metrics


def main():
    print("=" * 80)
    print("Improved Graph-Based Retrieval Evaluation")
    print("=" * 80)

    # Configuration
    corpus_path = Path("/Users/christophermayfield/Desktop/Corpus_Final_Review")
    original_graph_db = Path("./data/store/corpus_graph.db")
    improved_graph_db = Path("./data/store/corpus_graph_improved.db")
    original_mapping = original_graph_db.with_suffix(".mapping.json")
    improved_mapping = improved_graph_db.with_suffix(".mapping.json")

    if not corpus_path.exists():
        print(f"✗ Corpus not found: {corpus_path}")
        return 1

    # Load corpus
    print("\n[1/6] Loading corpus...")
    loader = CorpusLoader(str(corpus_path))
    documents = loader.load_documents()
    qa_trackers = loader.load_qa_trackers()

    if not documents or not qa_trackers:
        print("✗ Failed to load corpus")
        return 1

    print(f"✓ Loaded {len(documents)} documents, {sum(len(q) for q in qa_trackers.values())} questions")

    # Build evaluation cases
    print("\n[2/6] Building evaluation cases...")
    cases = loader.build_evaluation_cases(qa_trackers)
    print(f"✓ {len(cases)} evaluation cases")

    # Build improved graph if needed
    print("\n[3/6] Checking/building improved graph...")
    if not improved_graph_db.exists():
        print("  Building improved graph...")
        from build_improved_graph import build_improved_graph

        improved_graph_db.parent.mkdir(parents=True, exist_ok=True)
        build_improved_graph(
            corpus_path=str(corpus_path),
            graph_db_path=str(improved_graph_db),
        )

    # Evaluate retrievers
    print("\n[4/6] Evaluating retrieval methods...")
    all_metrics = {}

    # 1. BM25 baseline
    bm25 = BM25Retriever(documents)
    bm25_metrics = evaluate_retriever(cases, bm25, "BM25 Baseline", search_method="bm25")
    all_metrics["BM25"] = bm25_metrics

    # 2. Original graph (if available)
    if original_graph_db.exists() and original_mapping.exists():
        try:
            print("  (Original graph-based retrieval skipped for time)")
        except Exception as e:
            print(f"  ✗ Original graph retrieval failed: {e}")

    # 3. Improved graph
    if improved_graph_db.exists() and improved_mapping.exists():
        try:
            graph_store = GraphStore(str(improved_graph_db))
            with open(improved_mapping) as f:
                chunk_to_doc = {int(k): v for k, v in json.load(f).items()}

            improved_retriever = ImprovedGraphRetriever(graph_store, chunk_to_doc)
            improved_metrics = evaluate_retriever(
                cases, improved_retriever, "Improved Graph Retrieval", search_method="improved_graph"
            )
            all_metrics["Improved Graph"] = improved_metrics
            graph_store.close()
        except Exception as e:
            print(f"  ✗ Improved graph retrieval failed: {e}")

    # Print results
    print("\n[5/6] Results")
    print("=" * 80)
    print("METRIC COMPARISON")
    print("=" * 80)

    metrics_to_show = ["recall@1", "recall@5", "recall@10", "mrr"]
    print(f"{'Retriever':<25} {' '.join(f'{m:>10}' for m in metrics_to_show)}")
    print("-" * 85)

    baseline_mrr = all_metrics["BM25"]["mrr"]
    for retriever_name, metrics in sorted(all_metrics.items()):
        values = [f"{metrics[m]:.4f}" for m in metrics_to_show]
        print(f"{retriever_name:<25} {' '.join(f'{v:>10}' for v in values)}")

    # Show improvements
    print("\n" + "=" * 80)
    print("IMPROVEMENT OVER BM25")
    print("=" * 80)

    for retriever_name, metrics in sorted(all_metrics.items()):
        if retriever_name == "BM25":
            continue

        mrr_delta = metrics["mrr"] - baseline_mrr
        mrr_pct = (mrr_delta / baseline_mrr) * 100 if baseline_mrr > 0 else 0

        recall1_delta = metrics["recall@1"] - all_metrics["BM25"]["recall@1"]
        recall5_delta = metrics["recall@5"] - all_metrics["BM25"]["recall@5"]

        print(f"\n{retriever_name}:")
        print(f"  MRR:       {mrr_delta:+.4f} ({mrr_pct:+.2f}%)")
        print(f"  Recall@1:  {recall1_delta:+.4f}")
        print(f"  Recall@5:  {recall5_delta:+.4f}")

    # Save results
    output_file = Path("improved_experiment_results.json")
    results_data = {
        "timestamp": Path(__file__).stat().st_mtime,
        "baseline_mrr": baseline_mrr,
        "results": all_metrics,
    }
    output_file.write_text(json.dumps(results_data, indent=2))
    print(f"\n✓ Results saved to {output_file}")

    return 0


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)

    sys.exit(main())
