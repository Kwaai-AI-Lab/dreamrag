#!/usr/bin/env python3
"""
Run retrieval evaluation experiment to measure performance.

Compares BM25 baseline against previous results and tests with our fixes.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

from eval.schemas import RetrievalCase, RetrievalResult
from eval.retrieval_metrics import recall_at_k, mean_reciprocal_rank, ndcg_at_k
from eval.retrieval_rankers import BM25Retriever, reciprocal_rank_fusion
from corpus_retrieval_analysis import CorpusLoader


def main():
    print("=" * 80)
    print("DreamRAG Retrieval Evaluation Experiment")
    print("=" * 80)

    # Load corpus and QA data
    corpus_path = Path("/Users/christophermayfield/Desktop/Corpus_Final_Review")
    if not corpus_path.exists():
        print(f"✗ Corpus not found at {corpus_path}")
        return 1

    print("\n[1/3] Loading corpus and QA-Trackers...")
    loader = CorpusLoader(str(corpus_path))
    documents = loader.load_documents()
    qa_trackers = loader.load_qa_trackers()

    if not documents:
        print("✗ No documents loaded")
        return 1

    if not qa_trackers:
        print("✗ No QA trackers loaded")
        return 1

    print(f"\n✓ Loaded {len(documents)} documents and {sum(len(q) for q in qa_trackers.values())} questions")

    # Build evaluation cases
    print("\n[2/3] Building evaluation cases...")
    cases = loader.build_evaluation_cases(qa_trackers)
    print(f"✓ Built {len(cases)} evaluation cases")

    # Run BM25 retrieval
    print("\n[3/3] Running BM25 retrieval evaluation...")
    print(f"    Initializing BM25 retriever with {len(documents)} documents...")

    retriever = BM25Retriever(documents)

    results: List[RetrievalResult] = []
    for case in cases:
        result = retriever.search(
            query_id=case.query_id,
            query=case.query,
            top_k=10
        )
        results.append(result)

    print(f"✓ Completed {len(results)} retrieval queries")

    # Compute metrics
    print("\n[4/4] Computing retrieval metrics...")
    metrics = {
        "recall@1": recall_at_k(cases, results, k=1),
        "recall@3": recall_at_k(cases, results, k=3),
        "recall@5": recall_at_k(cases, results, k=5),
        "recall@10": recall_at_k(cases, results, k=10),
        "mrr": mean_reciprocal_rank(cases, results),
    }

    print("\n" + "=" * 80)
    print("RETRIEVAL PERFORMANCE RESULTS")
    print("=" * 80)

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:15} {metric_value:.4f}")

    # Compare with baseline
    print("\n" + "=" * 80)
    print("COMPARISON WITH PREVIOUS BASELINE")
    print("=" * 80)

    baseline_mrr = 0.6841  # From retrieval_analysis_results.json
    current_mrr = metrics["mrr"]
    mrr_delta = current_mrr - baseline_mrr
    mrr_pct = (mrr_delta / baseline_mrr) * 100 if baseline_mrr > 0 else 0

    print(f"Previous MRR:  {baseline_mrr:.4f}")
    print(f"Current MRR:   {current_mrr:.4f}")
    print(f"Delta:         {mrr_delta:+.4f} ({mrr_pct:+.2f}%)")
    print()

    if current_mrr > baseline_mrr:
        print(f"✓ IMPROVEMENT: Retrieval performance improved by {mrr_pct:.2f}%")
    elif current_mrr == baseline_mrr:
        print("= STABLE: Retrieval performance unchanged")
    else:
        print(f"⚠ REGRESSION: Retrieval performance decreased by {abs(mrr_pct):.2f}%")

    # Topic-level analysis
    print("\n" + "=" * 80)
    print("TOPIC-LEVEL ANALYSIS")
    print("=" * 80)

    topic_results: Dict[str, List[RetrievalResult]] = {}
    for result in results:
        topic = result.query_id.split("|")[0]
        topic_results.setdefault(topic, []).append(result)

    topic_cases: Dict[str, List[RetrievalCase]] = {}
    for case in cases:
        topic = case.query_id.split("|")[0]
        topic_cases.setdefault(topic, []).append(case)

    print(f"{'Topic':<50} {'Recall@5':>10} {'MRR':>10} {'Qs':>5}")
    print("-" * 78)

    for topic in sorted(topic_cases.keys()):
        topic_cases_list = topic_cases[topic]
        topic_results_list = topic_results[topic]

        recall = recall_at_k(topic_cases_list, topic_results_list, k=5)
        mrr = mean_reciprocal_rank(topic_cases_list, topic_results_list)

        topic_short = topic[:45]
        print(f"{topic_short:<50} {recall:>10.4f} {mrr:>10.4f} {len(topic_cases_list):>5}")

    # Save results
    output_file = Path("retrieval_experiment_results.json")
    results_data = {
        "timestamp": str(Path(__file__).stat().st_mtime),
        "summary": {
            "total_documents": len(documents),
            "total_questions": len(cases),
            "total_topics": len(topic_cases),
        },
        "metrics": metrics,
        "comparison": {
            "baseline_mrr": baseline_mrr,
            "current_mrr": current_mrr,
            "improvement": mrr_pct,
        },
        "topic_metrics": {},
    }

    for topic in sorted(topic_cases.keys()):
        topic_cases_list = topic_cases[topic]
        topic_results_list = topic_results[topic]
        results_data["topic_metrics"][topic] = {
            "num_questions": len(topic_cases_list),
            "recall@5": recall_at_k(topic_cases_list, topic_results_list, k=5),
            "mrr": mean_reciprocal_rank(topic_cases_list, topic_results_list),
        }

    output_file.write_text(json.dumps(results_data, indent=2))
    print(f"\n✓ Results saved to {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
