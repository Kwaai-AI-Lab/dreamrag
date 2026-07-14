"""
Compare retrieval metrics before and after PDF extraction
"""
import json

# Before (PDFs skipped)
before = {
    "total_documents": 85,
    "total_questions": 280,
    "recall@1": 0.268,
    "recall@5": 0.375,
    "recall@10": 0.425,
    "mrr": 0.316,
    "zero_recall_topics": 6,
    "zero_recall_questions": 120,
}

# Load new results when available
try:
    with open('retrieval_analysis_results.json', 'r') as f:
        results = json.load(f)
        after = {
            "total_documents": results['summary']['total_documents'],
            "total_questions": results['summary']['total_questions'],
            "recall@1": results['metrics']['recall@1'],
            "recall@5": results['metrics']['recall@5'],
            "recall@10": results['metrics']['recall@10'],
            "mrr": results['metrics']['mrr'],
        }

        # Count zero-recall topics
        zero_topics = sum(1 for t, m in results['topic_metrics'].items() if m['mrr'] == 0.0)
        zero_questions = zero_topics * 20
        after['zero_recall_topics'] = zero_topics
        after['zero_recall_questions'] = zero_questions

        print("=" * 80)
        print("BEFORE vs AFTER: PDF Extraction Impact")
        print("=" * 80)
        print()

        metrics = ['total_documents', 'recall@1', 'recall@5', 'recall@10', 'mrr']
        print(f"{'Metric':<20} {'Before':<15} {'After':<15} {'Change':<15}")
        print("-" * 65)

        for metric in metrics:
            before_val = before[metric]
            after_val = after[metric]

            if isinstance(before_val, int):
                change = after_val - before_val
                print(f"{metric:<20} {before_val:<15} {after_val:<15} +{change:<14}")
            else:
                change = (after_val - before_val) * 100
                print(f"{metric:<20} {before_val:<.3f}           {after_val:<.3f}           {change:+.1f}%")

        print()
        print(f"{'Zero-Recall Topics':<20} {before['zero_recall_topics']:<15} {after['zero_recall_topics']:<15}")
        print(f"{'Affected Questions':<20} {before['zero_recall_questions']:<15} {after['zero_recall_questions']:<15}")

        print()
        print("=" * 80)

except FileNotFoundError:
    print("Waiting for analysis results...")
