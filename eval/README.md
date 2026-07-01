# Dream RAG Evaluation Framework

## Overview

This directory contains the evaluation framework used to assess Dream RAG's retrieval performance, retrieval grounding, and graph refinement behavior.

Dream RAG is based on the hypothesis that iterative graph refinement ("dreaming") improves knowledge representation and retrieval quality over time.

The evaluation framework is designed to answer the following question:

> Does iterative graph refinement improve retrieval quality and retrieval grounding while maintaining an efficient and coherent graph structure?

---

# Evaluation Categories

The current evaluation framework consists of three categories:

## 1. Retrieval Effectiveness

Measures whether relevant information can be successfully retrieved.

Metrics:

* Recall@K
* Mean Reciprocal Rank (MRR)
* Normalized Discounted Cumulative Gain (nDCG)

Primary Question:

> Can the system retrieve the correct information?

---

## 2. Retrieval Grounding

Measures whether retrieved information is sufficient, relevant, and appropriately utilized.

Metrics:

* Context Recall
* Context Precision
* Faithfulness

Implementation:

* RAGAS Adapter
* RAGAS Runner

Primary Question:

> Is the generated answer supported by retrieved evidence?

---

## 3. Graph Refinement Quality

Measures how graph structure evolves through dream-cycle refinement.

Metrics:

* Entity Coverage
* Relation Completeness
* Compression Delta
* Deduplication Delta

Primary Question:

> Does graph refinement improve graph quality while maintaining retrieval performance?

---

# Directory Structure

```text
eval/
├── README.md
├── __init__.py
├── schemas.py
├── retrieval_metrics.py
├── graph_metrics.py
├── ragas_adapter.py
├── ragas_runner.py
└── future_work/
```

---

# Module Descriptions

## schemas.py

Shared evaluation data structures used throughout the evaluation framework.

## retrieval_metrics.py

Retrieval effectiveness metrics:

* Recall@K
* MRR
* nDCG

## graph_metrics.py

Dream RAG graph-refinement metrics:

* Entity Coverage
* Relation Completeness
* Compression Delta
* Deduplication Delta

## ragas_adapter.py

Converts Dream RAG evaluation records into a format compatible with RAGAS.

## ragas_runner.py

Executes retrieval-grounding evaluation using:

* Context Recall
* Context Precision
* Faithfulness

---

# Scope Boundaries

The current evaluation framework focuses on:

* Retrieval Effectiveness
* Retrieval Grounding
* Graph Refinement Quality

The following areas are intentionally excluded from the current scope:

* Security Evaluation
* Trustworthiness Evaluation
* Alignment and Safety Evaluation
* Comparative benchmarking against other RAG systems

These categories represent distinct research questions and are documented separately within the future_work directory.

---

# Future Evaluation Categories

Planned evaluation categories include:

## Trustworthiness

Potential dimensions:

* Groundedness
* Provenance
* Consistency
* Confidence Calibration

## Security

Potential dimensions:

* Knowledge Integrity
* Poison Persistence
* Recovery Rate
* Recovery Latency

## Alignment and Safety

Potential dimensions:

* Harmfulness
* Toxicity
* Refusal Quality
* Safety Compliance

These categories are intentionally deferred in order to maintain a focused evaluation of Dream RAG's retrieval and graph-refinement capabilities.
