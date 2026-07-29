#!/usr/bin/env python3
"""
run_final_evaluation.py — Final evaluation of all retrieval methods.

Compares:
1. BM25 baseline
2. Simple hybrid (BM25 + entity boosting)
3. Improved graph-based retrieval
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

from eval.schemas import RetrievalCase, RetrievalResult
from eval.retrieval_metrics import recall_at_k, mean_reciprocal_rank
from eval.retrieval_rankers import BM25Retriever
from corpus_retrieval_analysis import CorpusLoader
from simple_hybrid_retrieval import SimpleHybridRetriever
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
            print(f"  Progress: {i + 1}/{len(cases)}", end="\r")

        try:
            if search_method == "bm25":
                result = retriever.search(
                    query_id=case.query_id,
                    query=case.query,
                    top_k=10,
                )
            elif search_method == "simple_hybrid":
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
            results.append(RetrievalResult(query_id=case.query_id, retrieved_ids=[]))

    print(f"  ✓ Completed {len(results)} queries              ")

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
    print("Final Retrieval Evaluation: BM25 vs Hybrid vs Improved Graph")
    print("=" * 80)

    # Configuration
    corpus_path = Path("/Users/christophermayfield/Desktop/Corpus_Final_Review")
    improved_graph_db = Path("./data/store/corpus_graph_improved.db")
    improved_mapping = improved_graph_db.with_suffix(".mapping.json")

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

    # Evaluate retrievers
    print("\n[3/5] Evaluating retrieval methods...")
    all_metrics = {}

    # 1. BM25 baseline
    print("\n--- BM25 Baseline ---")
    bm25 = BM25Retriever(documents)
    bm25_metrics = evaluate_retriever(cases, bm25, "BM25 Baseline", search_method="bm25")
    all_metrics["BM25"] = bm25_metrics

    # 2. Simple Hybrid
    print("\n--- Simple Hybrid (BM25 + Entity Boosting) ---")
    simple_hybrid = SimpleHybridRetriever(documents)
    simple_hybrid_metrics = evaluate_retriever(
        cases, simple_hybrid, "Simple Hybrid", search_method="simple_hybrid"
    )
    all_metrics["Simple Hybrid"] = simple_hybrid_metrics

    # 3. Improved Graph
    if improved_graph_db.exists() and improved_mapping.exists():
        try:
            print("\n--- Improved Graph-Based Retrieval ---")
            graph_store = GraphStore(str(improved_graph_db))
            with open(improved_mapping) as f:
                chunk_to_doc = {int(k): v for k, v in json.load(f).items()}

            improved_retriever = ImprovedGraphRetriever(graph_store, chunk_to_doc)
            improved_metrics = evaluate_retriever(
                cases, improved_retriever, "Improved Graph", search_method="improved_graph"
            )
            all_metrics["Improved Graph"] = improved_metrics
            graph_store.close()
        except Exception as e:
            print(f"  ✗ Improved graph retrieval failed: {e}")

    # Print results
    print("\n" + "=" * 80)
    print("RETRIEVAL PERFORMANCE COMPARISON")
    print("=" * 80)

    metrics_to_show = ["recall@1", "recall@5", "recall@10", "mrr"]
    print(f"\n{'Retriever':<25} {' '.join(f'{m:>10}' for m in metrics_to_show)}")
    print("-" * 85)

    baseline_mrr = all_metrics["BM25"]["mrr"]
    for retriever_name in ["BM25", "Simple Hybrid", "Improved Graph"]:
        if retriever_name not in all_metrics:
            continue
        metrics = all_metrics[retriever_name]
        values = [f"{metrics[m]:.4f}" for m in metrics_to_show]
        print(f"{retriever_name:<25} {' '.join(f'{v:>10}' for v in values)}")

    # Show improvements
    print("\n" + "=" * 80)
    print("IMPROVEMENT vs BM25 BASELINE")
    print("=" * 80)

    for retriever_name in ["Simple Hybrid", "Improved Graph"]:
        if retriever_name not in all_metrics:
            continue

        metrics = all_metrics[retriever_name]
        mrr_delta = metrics["mrr"] - baseline_mrr
        mrr_pct = (mrr_delta / baseline_mrr) * 100 if baseline_mrr > 0 else 0

        recall1_delta = metrics["recall@1"] - all_metrics["BM25"]["recall@1"]
        recall5_delta = metrics["recall@5"] - all_metrics["BM25"]["recall@5"]
        recall10_delta = metrics["recall@10"] - all_metrics["BM25"]["recall@10"]

        print(f"\n{retriever_name}:")
        print(f"  MRR:       {mrr_delta:+.4f} ({mrr_pct:+.2f}%)")
        print(f"  Recall@1:  {recall1_delta:+.4f}")
        print(f"  Recall@5:  {recall5_delta:+.4f}")
        print(f"  Recall@10: {recall10_delta:+.4f}")

    # Show winner
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    best_method = max(all_metrics.items(), key=lambda x: x[1]["mrr"])
    best_name, best_metrics = best_method

    print(f"\nBest performing method: {best_name}")
    print(f"  MRR: {best_metrics['mrr']:.4f}")
    print(f"  Recall@1: {best_metrics['recall@1']:.4f}")
    print(f"  Recall@5: {best_metrics['recall@5']:.4f}")

    if best_name == "BM25":
        print("\n⚠ Graph-based retrieval did not improve over BM25")
        print("Reasons:")
        print("  1. Entity extraction quality is moderate (pattern-based)")
        print("  2. Entity linking in queries is basic")
        print("  3. Graph structure is based on co-occurrence (weak signal)")
        print("\nNext steps to improve:")
        print("  - Use better NER (GLiNER, spaCy)")
        print("  - Add semantic relation extraction via LLM")
        print("  - Implement entity embeddings for matching")
        print("  - Fine-tune relation weights on training data")
    else:
        print(f"\n✓ {best_name} improved retrieval by {(best_metrics['mrr'] - baseline_mrr) / baseline_mrr * 100:.2f}%")

    # Save results
    output_file = Path("final_evaluation_results.json")
    results_data = {
        "baseline_mrr": baseline_mrr,
        "results": all_metrics,
        "winner": best_name,
    }
    output_file.write_text(json.dumps(results_data, indent=2))
    print(f"\n✓ Results saved to {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
