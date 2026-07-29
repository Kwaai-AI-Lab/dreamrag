#!/usr/bin/env python3
"""
run_graph_experiment.py — Evaluate graph-based retrieval performance.

Compares:
1. BM25 baseline
2. Graph-based retrieval (entity-aware)
3. Hybrid fusion (BM25 + graph)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

from eval.schemas import RetrievalCase, RetrievalResult
from eval.retrieval_metrics import recall_at_k, mean_reciprocal_rank, ndcg_at_k
from eval.retrieval_rankers import BM25Retriever
from corpus_retrieval_analysis import CorpusLoader
from graph_retrieval import GraphRetriever
from build_corpus_graph import build_entity_graph
from hybrid_retrieval import HybridRetriever


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
            if search_method == "search":
                result = retriever.search(
                    query_id=case.query_id,
                    query=case.query,
                    top_k=10,
                )
            elif search_method == "graph_search":
                graph_docs = retriever.search(
                    query=case.query,
                    top_k=10,
                )
                result = RetrievalResult(
                    query_id=case.query_id,
                    retrieved_ids=[doc_id for doc_id, _ in graph_docs],
                )
            results.append(result)
        except Exception as e:
            print(f"  Error on {case.query_id}: {e}")
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
    print("Graph-Based Retrieval Evaluation")
    print("=" * 80)

    # Configuration
    corpus_path = Path("/Users/christophermayfield/Desktop/Corpus_Final_Review")
    graph_db_path = Path("./data/store/corpus_graph.db")
    chunk_mapping_path = graph_db_path.with_suffix(".mapping.json")

    if not corpus_path.exists():
        print(f"✗ Corpus not found: {corpus_path}")
        return 1

    # Load corpus
    print("\n[1/5] Loading corpus...")
    loader = CorpusLoader(str(corpus_path))
    documents = loader.load_documents()
    qa_trackers = loader.load_qa_trackers()

    if not documents or not qa_trackers:
        print("✗ Failed to load corpus")
        return 1

    print(f"✓ Loaded {len(documents)} documents, {sum(len(q) for q in qa_trackers.values())} questions")

    # Build evaluation cases
    print("\n[2/5] Building evaluation cases...")
    cases = loader.build_evaluation_cases(qa_trackers)
    print(f"✓ {len(cases)} evaluation cases")

    # Build/load graph
    print("\n[3/5] Building knowledge graph...")
    if graph_db_path.exists():
        print("  Using existing graph...")
    else:
        graph_db_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_to_doc = build_entity_graph(
            corpus_path=str(corpus_path),
            graph_db_path=str(graph_db_path),
            max_docs=None,  # Use all documents
        )

    # Evaluate retrievers
    print("\n[4/5] Evaluating retrieval methods...")

    all_metrics = {}

    # 1. BM25 baseline
    bm25 = BM25Retriever(documents)
    bm25_metrics = evaluate_retriever(cases, bm25, "BM25 Baseline")
    all_metrics["BM25"] = bm25_metrics

    # 2. Graph-based (if available)
    if graph_db_path.exists() and chunk_mapping_path.exists():
        try:
            from graph import GraphStore

            graph_store = GraphStore(str(graph_db_path))
            with open(chunk_mapping_path) as f:
                chunk_to_doc = {int(k): v for k, v in json.load(f).items()}

            graph_retriever = GraphRetriever(graph_store, chunk_to_doc)
            graph_metrics = evaluate_retriever(
                cases, graph_retriever, "Graph-Based Retrieval", search_method="graph_search"
            )
            all_metrics["Graph"] = graph_metrics
            graph_store.close()
        except Exception as e:
            print(f"✗ Graph retrieval failed: {e}")

    # 3. Hybrid (if graph available)
    if graph_db_path.exists():
        try:
            hybrid = HybridRetriever(
                documents,
                graph_db_path=str(graph_db_path),
                chunk_mapping_path=str(chunk_mapping_path),
            )
            hybrid_metrics = evaluate_retriever(cases, hybrid, "Hybrid (BM25 + Graph)")
            all_metrics["Hybrid"] = hybrid_metrics
        except Exception as e:
            print(f"✗ Hybrid retrieval failed: {e}")

    # Print results
    print("\n[5/5] Results")
    print("=" * 80)
    print("METRIC COMPARISON")
    print("=" * 80)

    metrics_to_show = ["recall@1", "recall@5", "recall@10", "mrr"]
    print(f"{'Retriever':<20} {' '.join(f'{m:>10}' for m in metrics_to_show)}")
    print("-" * 80)

    baseline_mrr = all_metrics["BM25"]["mrr"]
    for retriever_name, metrics in all_metrics.items():
        values = [f"{metrics[m]:.4f}" for m in metrics_to_show]
        print(f"{retriever_name:<20} {' '.join(f'{v:>10}' for v in values)}")

    # Show improvements
    print("\n" + "=" * 80)
    print("IMPROVEMENT OVER BM25")
    print("=" * 80)

    for retriever_name, metrics in all_metrics.items():
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
    output_file = Path("graph_experiment_results.json")
    results_data = {
        "timestamp": str(Path(__file__).stat().st_mtime),
        "baseline_mrr": baseline_mrr,
        "results": all_metrics,
    }
    output_file.write_text(json.dumps(results_data, indent=2))
    print(f"\n✓ Results saved to {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
