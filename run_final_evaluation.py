#!/usr/bin/env python3
"""
run_final_evaluation.py — Local retrieval evaluation.

Compares:
1. BM25 baseline
2. Simple hybrid (BM25 + entity boosting)
3. Improved graph-based retrieval (if graph DB + mapping exist)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from eval.schemas import RetrievalCase, RetrievalResult
from eval.retrieval_metrics import recall_at_k, mean_reciprocal_rank
from eval.retrieval_rankers import BM25Retriever
from corpus_retrieval_analysis import CorpusLoader
from simple_hybrid_retrieval import SimpleHybridRetriever
from improved_graph_retrieval import ImprovedGraphRetriever
from graph import GraphStore
from gliner_ner import DEFAULT_URL

DEFAULT_CORPUS = Path("/Users/christophermayfield/Desktop/Corpus_Final_Review")
DEFAULT_GRAPH_DB = Path("./data/store/corpus_graph_improved.db")
DEFAULT_OUT = Path("final_evaluation_results.json")


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
            if search_method in ("bm25", "simple_hybrid"):
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
            else:
                result = RetrievalResult(query_id=case.query_id, retrieved_ids=[])
            results.append(result)
        except Exception:
            results.append(RetrievalResult(query_id=case.query_id, retrieved_ids=[]))

    print(f"  ✓ Completed {len(results)} queries              ")

    return {
        "recall@1": recall_at_k(cases, results, k=1),
        "recall@3": recall_at_k(cases, results, k=3),
        "recall@5": recall_at_k(cases, results, k=5),
        "recall@10": recall_at_k(cases, results, k=10),
        "mrr": mean_reciprocal_rank(cases, results),
    }


def run_evaluation(
    corpus_path: Path,
    graph_db: Path = DEFAULT_GRAPH_DB,
    out_path: Path = DEFAULT_OUT,
    gliner_url: Optional[str] = DEFAULT_URL,
    skip_graph: bool = False,
) -> int:
    """Run BM25 / hybrid / graph eval and write JSON results."""
    print("=" * 80)
    print("Local Retrieval Evaluation: BM25 vs Hybrid vs Improved Graph")
    print("=" * 80)

    mapping_path = graph_db.with_suffix(".mapping.json")

    if not corpus_path.exists():
        print(f"✗ Corpus not found: {corpus_path}")
        return 1

    print("\n[1/5] Loading corpus...")
    loader = CorpusLoader(str(corpus_path))
    documents = loader.load_documents()
    qa_trackers = loader.load_qa_trackers()

    if not documents or not qa_trackers:
        print("✗ Failed to load corpus")
        return 1

    print(
        f"✓ Loaded {len(documents)} documents, "
        f"{sum(len(q) for q in qa_trackers.values())} questions"
    )

    print("\n[2/5] Building evaluation cases...")
    cases = loader.build_evaluation_cases(qa_trackers)
    print(f"✓ {len(cases)} evaluation cases")

    print("\n[3/5] Evaluating retrieval methods...")
    all_metrics: Dict[str, Dict] = {}

    print("\n--- BM25 Baseline ---")
    bm25 = BM25Retriever(documents)
    all_metrics["BM25"] = evaluate_retriever(
        cases, bm25, "BM25 Baseline", search_method="bm25"
    )

    print("\n--- Simple Hybrid (BM25 + Entity Boosting) ---")
    simple_hybrid = SimpleHybridRetriever(documents, gliner_url=gliner_url)
    all_metrics["Simple Hybrid"] = evaluate_retriever(
        cases, simple_hybrid, "Simple Hybrid", search_method="simple_hybrid"
    )

    if not skip_graph and graph_db.exists() and mapping_path.exists():
        try:
            print("\n--- Improved Graph-Based Retrieval ---")
            graph_store = GraphStore(str(graph_db))
            with open(mapping_path) as f:
                chunk_to_doc = {int(k): v for k, v in json.load(f).items()}

            improved_retriever = ImprovedGraphRetriever(
                graph_store, chunk_to_doc, gliner_url=gliner_url
            )
            all_metrics["Improved Graph"] = evaluate_retriever(
                cases, improved_retriever, "Improved Graph", search_method="improved_graph"
            )
            graph_store.close()
        except Exception as e:
            print(f"  ✗ Improved graph retrieval failed: {e}")
    elif not skip_graph:
        print(
            f"\n⚠ Skipping Improved Graph "
            f"(missing {graph_db} and/or {mapping_path})"
        )

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

    print("\n" + "=" * 80)
    print("IMPROVEMENT vs BM25 BASELINE")
    print("=" * 80)

    for retriever_name in ["Simple Hybrid", "Improved Graph"]:
        if retriever_name not in all_metrics:
            continue
        metrics = all_metrics[retriever_name]
        mrr_delta = metrics["mrr"] - baseline_mrr
        mrr_pct = (mrr_delta / baseline_mrr) * 100 if baseline_mrr > 0 else 0
        print(f"\n{retriever_name}:")
        print(f"  MRR:       {mrr_delta:+.4f} ({mrr_pct:+.2f}%)")
        print(f"  Recall@1:  {metrics['recall@1'] - all_metrics['BM25']['recall@1']:+.4f}")
        print(f"  Recall@5:  {metrics['recall@5'] - all_metrics['BM25']['recall@5']:+.4f}")
        print(f"  Recall@10: {metrics['recall@10'] - all_metrics['BM25']['recall@10']:+.4f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    best_name, best_metrics = max(all_metrics.items(), key=lambda x: x[1]["mrr"])
    print(f"\nBest performing method: {best_name}")
    print(f"  MRR: {best_metrics['mrr']:.4f}")
    print(f"  Recall@1: {best_metrics['recall@1']:.4f}")
    print(f"  Recall@5: {best_metrics['recall@5']:.4f}")

    if best_name != "BM25":
        print(
            f"\n✓ {best_name} improved retrieval by "
            f"{(best_metrics['mrr'] - baseline_mrr) / baseline_mrr * 100:.2f}%"
        )

    results_data = {
        "baseline_mrr": baseline_mrr,
        "results": all_metrics,
        "winner": best_name,
        "corpus_path": str(corpus_path),
        "graph_db": str(graph_db),
    }
    out_path = Path(out_path)
    out_path.write_text(json.dumps(results_data, indent=2))
    print(f"\n✓ Results saved to {out_path}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Local BM25 / hybrid / graph evaluation")
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=DEFAULT_CORPUS,
        help="Path to Corpus_Final_Review (or compatible corpus)",
    )
    parser.add_argument(
        "--graph-db",
        type=Path,
        default=DEFAULT_GRAPH_DB,
        help="Improved graph SQLite DB",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSON path",
    )
    parser.add_argument(
        "--gliner-url",
        default=DEFAULT_URL,
        help="GLiNER NER server URL (used by hybrid / graph)",
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Only run BM25 and Simple Hybrid",
    )
    args = parser.parse_args(argv)
    return run_evaluation(
        corpus_path=args.corpus_path,
        graph_db=args.graph_db,
        out_path=args.out,
        gliner_url=args.gliner_url or None,
        skip_graph=args.skip_graph,
    )


if __name__ == "__main__":
    sys.exit(main())
