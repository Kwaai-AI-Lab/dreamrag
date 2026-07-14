"""
Corpus Retrieval Analysis for DreamRAG
Evaluates retrieval performance against Corpus_Final_Review using QA-Trackers
"""
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Set
import openpyxl
import re

# Suppress pdfplumber FontBBox warnings
warnings.filterwarnings('ignore', message='Could not get FontBBox')

from eval.schemas import RetrievalCase, RetrievalResult
from eval.retrieval_metrics import recall_at_k, mean_reciprocal_rank, ndcg_at_k
from eval.retrieval_rankers import BM25Retriever


class CorpusLoader:
    """Load documents and QA-Trackers from Corpus_Final_Review folder"""

    def __init__(self, corpus_path: str):
        self.corpus_path = Path(corpus_path)
        self.documents: Dict[str, str] = {}
        self.topics: List[str] = []

    def load_documents(self) -> Dict[str, str]:
        """Load all documents from the corpus structure"""
        doc_count = 0
        for topic_dir in sorted(self.corpus_path.iterdir()):
            if not topic_dir.is_dir() or topic_dir.name.startswith('.'):
                continue
            if topic_dir.name == "QA-Trackers":
                continue

            self.topics.append(topic_dir.name)
            docs_dir = topic_dir / "documents"

            if not docs_dir.exists():
                continue

            for doc_file in sorted(docs_dir.iterdir()):
                if doc_file.name.startswith('.'):
                    continue

                # Create unique doc_id from topic and filename
                doc_id = f"{topic_dir.name}|{doc_file.name}"
                try:
                    text = self._read_file(doc_file)
                    if text:
                        self.documents[doc_id] = text
                        doc_count += 1
                except Exception as e:
                    print(f"  Error reading {doc_file}: {e}")

        print(f"Loaded {doc_count} documents across {len(self.topics)} topics")
        return self.documents

    def _read_file(self, file_path: Path) -> str:
        """Read document content based on file type"""
        if file_path.suffix == ".pdf":
            try:
                import pdfplumber
                text = ""
                try:
                    with pdfplumber.open(file_path) as pdf:
                        # Limit to first 5 pages to avoid slow PDFs
                        for page in pdf.pages[:5]:
                            try:
                                page_text = page.extract_text()
                                if page_text:
                                    text += page_text + "\n"
                            except Exception:
                                continue
                        return text[:10000]  # Limit to 10K chars per doc
                except Exception as e:
                    # Skip problematic PDFs silently
                    return ""
            except ImportError:
                print(f"  pdfplumber not available, skipping {file_path.name}")
                return ""
        elif file_path.suffix == ".txt":
            return file_path.read_text(errors='ignore')[:10000]
        elif file_path.suffix == ".html":
            text = file_path.read_text(errors='ignore')
            # Simple HTML stripping
            text = re.sub(r'<[^>]+>', '', text)
            return text[:10000]
        elif file_path.suffix == ".vtt":
            # WebVTT transcript format
            text = file_path.read_text(errors='ignore')
            # Remove VTT headers and timing info
            lines = [
                line for line in text.split('\n')
                if line and not line.startswith('WEBVTT') and '-->' not in line
            ]
            return ' '.join(lines)[:10000]
        else:
            return ""

    def load_qa_trackers(self) -> Dict[str, Dict]:
        """Load all QA-Tracker xlsx files and extract questions with relevant docs"""
        trackers = {}
        qa_dir = self.corpus_path / "QA-Trackers"

        for tracker_file in sorted(qa_dir.glob("*.xlsx")):
            topic_name = tracker_file.stem  # Filename without .xlsx
            # Normalize topic name to match folder names
            topic_name = topic_name.replace('_India-focused_', ' (India-focused)')
            topic_name = topic_name.replace('_', ' ')

            try:
                wb = openpyxl.load_workbook(tracker_file)
                ws = wb.active
                questions = []

                # Row 1 is headers, data starts at row 2
                for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
                    if not row[0]:  # Empty question
                        continue

                    question = row[0]
                    # NotebookLM answer in column B, OpenWebUI in column C
                    nb_answer = row[1] if len(row) > 1 else ""

                    questions.append({
                        'query_id': f"{topic_name}|q{row_idx}",
                        'query': question,
                        'nb_answer': nb_answer,
                    })

                if questions:
                    trackers[topic_name] = questions
                    print(f"  Loaded {len(questions)} questions from {topic_name}")

            except Exception as e:
                print(f"  Error loading {tracker_file}: {e}")

        return trackers

    def build_evaluation_cases(
        self, trackers: Dict[str, Dict], doc_mapping: Dict[str, str] = None
    ) -> List[RetrievalCase]:
        """
        Build RetrievalCase objects from QA trackers.

        Maps questions to relevant documents based on topic matching.
        Since the corpus doesn't have explicit doc-to-question mappings,
        we use topic-level grouping: all docs in a topic are considered relevant
        for questions from that topic.
        """
        if doc_mapping is None:
            # Build implicit mapping: all docs from topic X are relevant for Q from topic X
            doc_mapping = {}
            for doc_id in self.documents.keys():
                topic = doc_id.split('|')[0]
                if topic not in doc_mapping:
                    doc_mapping[topic] = set()
                doc_mapping[topic].add(doc_id)

        cases = []
        for topic_name, questions in trackers.items():
            relevant_ids = doc_mapping.get(topic_name, set())

            for q_data in questions:
                case = RetrievalCase(
                    query_id=q_data['query_id'],
                    query=q_data['query'],
                    relevant_ids=relevant_ids,
                )
                cases.append(case)

        print(f"\nBuilt {len(cases)} evaluation cases")
        return cases


class RetrievalAnalyzer:
    """Run and analyze retrieval performance"""

    def __init__(self, documents: Dict[str, str], cases: List[RetrievalCase]):
        self.documents = documents
        self.cases = cases
        self.retriever = BM25Retriever(documents)
        self.results: List[RetrievalResult] = []

    def run_retrieval(self, top_k: int = 10) -> List[RetrievalResult]:
        """Run BM25 retrieval for all test cases"""
        print(f"\nRunning BM25 retrieval (top-{top_k})...")

        for case in self.cases:
            result = self.retriever.search(case.query_id, case.query, top_k=top_k)
            self.results.append(result)

        print(f"Retrieved results for {len(self.results)} queries")
        return self.results

    def compute_metrics(self, k_values: List[int] = None) -> Dict:
        """Compute retrieval metrics at different K values"""
        if k_values is None:
            k_values = [1, 3, 5, 10]

        metrics = {}

        # Recall@K
        for k in k_values:
            recall = recall_at_k(self.cases, self.results, k)
            metrics[f'recall@{k}'] = recall
            print(f"  Recall@{k}: {recall:.3f}")

        # MRR
        mrr = mean_reciprocal_rank(self.cases, self.results)
        metrics['mrr'] = mrr
        print(f"  MRR: {mrr:.3f}")

        # nDCG@K (use binary relevance: in topic = 1, out = 0)
        relevance_map = self._build_relevance_map()
        for k in k_values:
            ndcg = ndcg_at_k(relevance_map, self.results, k)
            metrics[f'ndcg@{k}'] = ndcg
            print(f"  nDCG@{k}: {ndcg:.3f}")

        return metrics

    def _build_relevance_map(self) -> Dict[str, Dict[str, float]]:
        """Build relevance map for nDCG: query_id -> {doc_id -> score}"""
        relevance_map = {}
        for case in self.cases:
            relevance_map[case.query_id] = {
                doc_id: 1.0 if doc_id in case.relevant_ids else 0.0
                for doc_id in self.documents.keys()
            }
        return relevance_map

    def analyze_by_topic(self) -> Dict:
        """Break down metrics by topic"""
        topic_results = {}

        # Group cases and results by topic
        cases_by_topic = {}
        results_by_topic = {}

        for case in self.cases:
            topic = case.query_id.split('|')[0]
            if topic not in cases_by_topic:
                cases_by_topic[topic] = []
            cases_by_topic[topic].append(case)

        for result in self.results:
            topic = result.query_id.split('|')[0]
            if topic not in results_by_topic:
                results_by_topic[topic] = []
            results_by_topic[topic].append(result)

        # Compute metrics per topic
        for topic in sorted(cases_by_topic.keys()):
            cases = cases_by_topic[topic]
            results = results_by_topic.get(topic, [])

            if not cases or not results:
                continue

            topic_metrics = {
                'num_questions': len(cases),
                'recall@1': recall_at_k(cases, results, 1),
                'recall@5': recall_at_k(cases, results, 5),
                'mrr': mean_reciprocal_rank(cases, results),
            }
            topic_results[topic] = topic_metrics

        return topic_results

    def get_sample_results(self, num_samples: int = 3) -> List[Dict]:
        """Get sample retrieval results for inspection"""
        samples = []

        for i, result in enumerate(self.results[:num_samples]):
            case = next((c for c in self.cases if c.query_id == result.query_id), None)
            if not case:
                continue

            retrieved_docs = []
            for doc_id in result.retrieved_ids:
                is_relevant = doc_id in case.relevant_ids
                topic = doc_id.split('|')[0]
                filename = doc_id.split('|')[1]
                retrieved_docs.append({
                    'topic': topic,
                    'filename': filename,
                    'relevant': is_relevant,
                })

            samples.append({
                'query_id': result.query_id,
                'query': case.query[:100] + '...' if len(case.query) > 100 else case.query,
                'num_relevant_in_topic': len(case.relevant_ids),
                'retrieved': retrieved_docs,
            })

        return samples


def main():
    corpus_path = "/Users/christophermayfield/Desktop/Corpus_Final_Review"

    print("=" * 80)
    print("DREAMRAG CORPUS RETRIEVAL ANALYSIS")
    print("=" * 80)

    # Load corpus
    print("\n[1] Loading Corpus...")
    loader = CorpusLoader(corpus_path)
    documents = loader.load_documents()
    trackers = loader.load_qa_trackers()

    # Build evaluation cases
    print("\n[2] Building Evaluation Cases...")
    cases = loader.build_evaluation_cases(trackers)

    # Run retrieval
    print("\n[3] Running Retrieval...")
    analyzer = RetrievalAnalyzer(documents, cases)
    analyzer.run_retrieval(top_k=10)

    # Compute metrics
    print("\n[4] Computing Metrics...")
    metrics = analyzer.compute_metrics()

    # Per-topic analysis
    print("\n[5] Topic-Level Analysis...")
    topic_metrics = analyzer.analyze_by_topic()
    print("\nMetrics by Topic:")
    print("-" * 80)
    for topic in sorted(topic_metrics.keys()):
        m = topic_metrics[topic]
        print(f"{topic:45} | Q:{m['num_questions']:2d} | R@1:{m['recall@1']:.2f} | "
              f"R@5:{m['recall@5']:.2f} | MRR:{m['mrr']:.3f}")

    # Sample results
    print("\n[6] Sample Results...")
    samples = analyzer.get_sample_results(3)
    for sample in samples:
        print(f"\nQuery: {sample['query']}")
        print(f"  Relevant docs in topic: {sample['num_relevant_in_topic']}")
        print("  Retrieved:")
        for doc in sample['retrieved'][:5]:
            status = "✓" if doc['relevant'] else "✗"
            print(f"    {status} {doc['topic']:30} - {doc['filename'][:40]}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Documents: {len(documents)}")
    print(f"Total Questions: {len(cases)}")
    print(f"Total Topics: {len(loader.topics)}")
    print(f"\nOverall Metrics:")
    for k in [1, 5, 10]:
        print(f"  Recall@{k}: {metrics.get(f'recall@{k}', 0):.3f}")
    print(f"  MRR: {metrics['mrr']:.3f}")

    # Save detailed results
    results_file = "retrieval_analysis_results.json"
    output_data = {
        'summary': {
            'total_documents': len(documents),
            'total_questions': len(cases),
            'total_topics': len(loader.topics),
        },
        'metrics': metrics,
        'topic_metrics': topic_metrics,
    }

    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nDetailed results saved to {results_file}")


if __name__ == "__main__":
    main()
