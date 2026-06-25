# Dream RAG Evaluation Metrics Rationale

## Recall@K

| Attribute              | Description                                                                      |
| ---------------------- | -------------------------------------------------------------------------------- |
| Purpose                | Measures whether relevant information appears within the top K retrieved results |
| Relevance to Dream RAG | Tests whether graph refinement improves retrieval effectiveness over time        |
| Evaluation Scope       | Retrieval Quality                                                                |
| Evaluation Points      | K = 1, 5, 10                                                                     |
| Future Extensions      | Longitudinal analysis across dream cycles and graph revisions                    |

---

## Mean Reciprocal Rank (MRR)

| Attribute              | Description                                                                 |
| ---------------------- | --------------------------------------------------------------------------- |
| Purpose                | Measures how highly ranked the first correct retrieval appears              |
| Relevance to Dream RAG | Evaluates whether dream cycles improve the ordering of relevant information |
| Evaluation Scope       | Retrieval Quality                                                           |
| Evaluation Points      | MRR across benchmark queries                                                |
| Future Extensions      | Analyze ranking stability across graph refinement iterations                |

---

## Normalized Discounted Cumulative Gain (nDCG)

| Attribute              | Description                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------- |
| Purpose                | Measures ranked retrieval quality while accounting for varying relevance levels                   |
| Relevance to Dream RAG | Determines whether graph refinement surfaces more useful information earlier in retrieval results |
| Evaluation Scope       | Retrieval Quality                                                                                 |
| Evaluation Points      | nDCG@K                                                                                            |
| Future Extensions      | Evaluate ranking quality throughout graph evolution                                               |

---

## RAGAS Context Recall

| Attribute              | Description                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------- |
| Purpose                | Measures whether retrieved context contains the information necessary to answer a query correctly |
| Relevance to Dream RAG | Evaluates whether dream cycles improve retrieval completeness                                     |
| Evaluation Scope       | Retrieval Quality                                                                                 |
| Evaluation Points      | Context Recall score                                                                              |
| Future Extensions      | Longitudinal retrieval completeness analysis                                                      |

---

## RAGAS Context Precision

| Attribute              | Description                                                                  |
| ---------------------- | ---------------------------------------------------------------------------- |
| Purpose                | Measures how much of the retrieved context is actually relevant to the query |
| Relevance to Dream RAG | Evaluates whether graph refinement reduces retrieval noise                   |
| Evaluation Scope       | Retrieval Quality                                                            |
| Evaluation Points      | Context Precision score                                                      |
| Future Extensions      | Assess retrieval efficiency as graph complexity changes                      |

---

## RAGAS Faithfulness

| Attribute              | Description                                                                    |
| ---------------------- | ------------------------------------------------------------------------------ |
| Purpose                | Measures whether generated answers are supported by retrieved evidence         |
| Relevance to Dream RAG | Evaluates whether improved retrieval translates into better-grounded responses |
| Evaluation Scope       | Retrieval Quality                                                              |
| Evaluation Points      | Faithfulness score                                                             |
| Future Extensions      | Support future reliability and trustworthiness assessments                     |

---

## Entity Coverage

| Attribute              | Description                                                                         |
| ---------------------- | ----------------------------------------------------------------------------------- |
| Purpose                | Measures the proportion of expected entities represented within the knowledge graph |
| Relevance to Dream RAG | Directly evaluates graph expansion and completion through dream cycles              |
| Evaluation Scope       | Graph Refinement                                                                    |
| Evaluation Points      | Covered entities / expected entities                                                |
| Future Extensions      | Analyze graph completeness growth over time                                         |

---

## Relation Completeness

| Attribute              | Description                                                                    |
| ---------------------- | ------------------------------------------------------------------------------ |
| Purpose                | Measures the proportion of expected relationships represented within the graph |
| Relevance to Dream RAG | Evaluates whether dreaming improves graph connectivity and semantic structure  |
| Evaluation Scope       | Graph Refinement                                                               |
| Evaluation Points      | Valid relations / expected relations                                           |
| Future Extensions      | Analyze ontology alignment and structural maturation                           |

---

## Compression Delta

| Attribute              | Description                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------- |
| Purpose                | Measures graph size reduction following consolidation, merging, and pruning operations |
| Relevance to Dream RAG | Tests the memory-consolidation hypothesis central to the Dream RAG architecture        |
| Evaluation Scope       | Graph Refinement                                                                       |
| Evaluation Points      | Nodes and edges before vs. after refinement                                            |
| Future Extensions      | Study efficiency-performance tradeoffs during graph evolution                          |

---

## Deduplication Delta

| Attribute              | Description                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------- |
| Purpose                | Measures the effect of duplicate removal on graph structure and retrieval performance |
| Relevance to Dream RAG | Evaluates whether graph simplification can occur without degrading retrieval quality  |
| Evaluation Scope       | Graph Refinement                                                                      |
| Evaluation Points      | Duplicate entities/relations removed and corresponding retrieval impact               |
| Future Extensions      | Analyze long-term graph maintenance and self-optimization behavior                    |

---

# Deferred Evaluation Categories

The following evaluation categories are intentionally outside the current retrieval-quality and graph-refinement scope.

## Trustworthiness Evaluation

Potential evaluation dimensions:

* Groundedness
* Provenance
* Consistency
* Confidence Calibration

---

## Security Evaluation

Potential evaluation dimensions:

* Knowledge Integrity
* Poison Persistence
* Recovery Rate
* Recovery Latency
* Attack Success Rate

---

## Alignment and Safety Evaluation

Potential evaluation dimensions:

* Harmfulness
* Toxicity
* Refusal Quality
* Safety Compliance
* Constitutional Adherence

These categories may be explored in future evaluation efforts as Dream RAG capabilities mature.



# Dream RAG Evaluation Metrics Rationale

## Purpose

This document defines the evaluation metrics selected for Dream RAG and explains how each metric contributes to assessing retrieval quality and graph refinement performance.

The current evaluation scope focuses on:

1. Retrieval Effectiveness
2. Retrieval Grounding
3. Graph Refinement Quality

The goal is to evaluate whether iterative graph refinement ("dreaming") improves retrieval quality over time.

The current scope intentionally excludes:

* Security evaluation
* Trustworthiness evaluation
* Alignment and safety evaluation
* Comparative benchmarking against other RAG systems

These areas may be explored through future evaluation efforts but are outside the scope of the current retrieval-quality assessment.

---

# Retrieval Effectiveness Metrics

## Recall@K

| Attribute              | Description                                                                      |
| ---------------------- | -------------------------------------------------------------------------------- |
| Purpose                | Measures whether relevant information appears within the top K retrieved results |
| Relevance to Dream RAG | Tests whether graph refinement improves retrieval effectiveness over time        |
| Evaluation Scope       | Retrieval Quality                                                                |
| Evaluation Points      | K = 1, 5, 10                                                                     |

### Rationale

Recall@K directly measures the ability of the system to retrieve relevant information. Improvements across dream cycles provide evidence that graph refinement enhances retrieval effectiveness.

---

## Mean Reciprocal Rank (MRR)

| Attribute              | Description                                                                 |
| ---------------------- | --------------------------------------------------------------------------- |
| Purpose                | Measures how highly ranked the first correct retrieval appears              |
| Relevance to Dream RAG | Evaluates whether dream cycles improve the ordering of relevant information |
| Evaluation Scope       | Retrieval Quality                                                           |
| Evaluation Points      | MRR across benchmark queries                                                |

### Rationale

Retrieving the correct information is valuable, but retrieving it earlier is better. MRR measures whether graph refinement improves retrieval ranking quality.

---

## Normalized Discounted Cumulative Gain (nDCG)

| Attribute              | Description                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------- |
| Purpose                | Measures ranked retrieval quality while accounting for varying relevance levels                   |
| Relevance to Dream RAG | Determines whether graph refinement surfaces more useful information earlier in retrieval results |
| Evaluation Scope       | Retrieval Quality                                                                                 |
| Evaluation Points      | nDCG@K                                                                                            |

### Rationale

nDCG provides a ranking-sensitive measure of retrieval quality and complements Recall@K and MRR by evaluating the overall usefulness of retrieval ordering.

---

# Retrieval Grounding Metrics

## RAGAS Context Recall

| Attribute              | Description                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------- |
| Purpose                | Measures whether retrieved context contains the information necessary to answer a query correctly |
| Relevance to Dream RAG | Evaluates whether dream cycles improve retrieval completeness                                     |
| Evaluation Scope       | Retrieval Grounding                                                                               |
| Evaluation Points      | Context Recall score                                                                              |

### Rationale

A retrieval system may retrieve documents while still missing critical information. Context Recall evaluates whether the retrieved evidence is sufficient for answering the query.

---

## RAGAS Context Precision

| Attribute              | Description                                                                  |
| ---------------------- | ---------------------------------------------------------------------------- |
| Purpose                | Measures how much of the retrieved context is actually relevant to the query |
| Relevance to Dream RAG | Evaluates whether graph refinement reduces retrieval noise                   |
| Evaluation Scope       | Retrieval Grounding                                                          |
| Evaluation Points      | Context Precision score                                                      |

### Rationale

Higher Context Precision indicates that retrieved evidence is focused and relevant rather than noisy or redundant.

---

## RAGAS Faithfulness

| Attribute              | Description                                                                    |
| ---------------------- | ------------------------------------------------------------------------------ |
| Purpose                | Measures whether generated answers are supported by retrieved evidence         |
| Relevance to Dream RAG | Evaluates whether improved retrieval translates into better-grounded responses |
| Evaluation Scope       | Retrieval Grounding                                                            |
| Evaluation Points      | Faithfulness score                                                             |

### Rationale

Faithfulness evaluates whether retrieved knowledge is actually being used correctly by downstream generation systems.

---

# Graph Refinement Metrics

## Entity Coverage

| Attribute              | Description                                                                         |
| ---------------------- | ----------------------------------------------------------------------------------- |
| Purpose                | Measures the proportion of expected entities represented within the knowledge graph |
| Relevance to Dream RAG | Directly evaluates graph expansion and completion through dream cycles              |
| Evaluation Scope       | Graph Refinement                                                                    |
| Evaluation Points      | Covered entities / expected entities                                                |

### Rationale

Entity Coverage measures graph completeness and provides evidence that iterative refinement expands knowledge representation.

---

## Relation Completeness

| Attribute              | Description                                                                    |
| ---------------------- | ------------------------------------------------------------------------------ |
| Purpose                | Measures the proportion of expected relationships represented within the graph |
| Relevance to Dream RAG | Evaluates whether dreaming improves graph connectivity and semantic structure  |
| Evaluation Scope       | Graph Refinement                                                               |
| Evaluation Points      | Valid relations / expected relations                                           |

### Rationale

Knowledge retrieval depends on relationships as much as entities. Relation Completeness evaluates structural graph quality.

---

## Compression Delta

| Attribute              | Description                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------- |
| Purpose                | Measures graph size reduction following consolidation, merging, and pruning operations |
| Relevance to Dream RAG | Tests the memory-consolidation hypothesis central to the Dream RAG architecture        |
| Evaluation Scope       | Graph Refinement                                                                       |
| Evaluation Points      | Nodes and edges before versus after refinement                                         |

### Rationale

Dream RAG proposes that knowledge can be refined and compressed without sacrificing retrieval performance. Compression Delta measures this behavior directly.

---

## Deduplication Delta

| Attribute              | Description                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------- |
| Purpose                | Measures the effect of duplicate removal on graph structure and retrieval performance |
| Relevance to Dream RAG | Evaluates whether graph simplification can occur without degrading retrieval quality  |
| Evaluation Scope       | Graph Refinement                                                                      |
| Evaluation Points      | Duplicate entities and relations removed and corresponding retrieval impact           |

### Rationale

Deduplication Delta evaluates whether memory consolidation reduces graph complexity while preserving useful knowledge.

---

# Deferred Evaluation Categories

The following evaluation categories are intentionally excluded from the current retrieval-quality and graph-refinement scope.

Their inclusion would introduce additional research questions beyond the objectives of the present evaluation.

## Trustworthiness Evaluation

Potential evaluation dimensions:

* Groundedness
* Provenance
* Consistency
* Confidence Calibration

### Research Focus

Assessment of reliability, trustworthiness, and confidence in knowledge retrieval systems.

---

## Security Evaluation

Potential evaluation dimensions:

* Knowledge Integrity
* Poison Persistence
* Recovery Rate
* Recovery Latency
* Attack Success Rate

### Research Focus

Assessment of adversarial robustness, graph corruption resilience, and knowledge-base security.

---

## Alignment and Safety Evaluation

Potential evaluation dimensions:

* Harmfulness
* Toxicity
* Refusal Quality
* Safety Compliance
* Constitutional Adherence

### Research Focus

Assessment of behavioral reliability, safety, and alignment characteristics.

---

# Summary

The current evaluation suite is designed to answer a single research question:

> Does iterative graph refinement improve retrieval quality over time?

To answer this question, Dream RAG is evaluated across:

* Retrieval Effectiveness
* Retrieval Grounding
* Graph Refinement Quality

Additional trust, security, and alignment evaluations remain outside the current scope and may be explored through future evaluation efforts.
