# DreamRAG Graph Retrieval Improvements - TODO

## Current State
- ✅ Graph infrastructure: 2,643 entities, 43,188 relations
- ✅ Graph retrieval pipeline: fully operational
- ✅ Simple hybrid approach: +0.22% MRR improvement
- ⚠️ Complex graph-based retrieval: -92.7% performance (not ready)
- 🎯 Baseline BM25 MRR: 0.6841

## Goal
Improve retrieval performance to **+15-30% MRR over BM25** through better entity/relation extraction and hybrid fusion.

---

## Phase 1: Improve Entity Extraction (Est. +5-10% MRR)

### Task 1.1: Integrate GLiNER for High-Quality NER
- **Priority:** HIGH
- **Effort:** 4-6 hours
- **Impact:** +5-10% MRR
- **Description:** 
  - Deploy GLiNER server (check `scripts/gliner_server.py`)
  - Create client wrapper to call GLiNER for entity extraction
  - Replace pattern-based extraction with GLiNER calls
  - Benchmark NER precision on sample documents
- **Success Criteria:**
  - NER precision ≥80% on test sample
  - Integration with graph building pipeline
  - Performance impact measured

### Task 1.2: Add spaCy Pipeline When Available
- **Priority:** MEDIUM
- **Effort:** 2-3 hours
- **Impact:** +2-3% MRR
- **Description:**
  - Try spaCy if installed, fall back to patterns
  - Use official entity types (PERSON, ORG, GPE, etc.)
  - Better name normalization
- **Success Criteria:**
  - Detects more entities than pattern-based
  - No performance regression

### Task 1.3: Implement Entity Embeddings for Linking
- **Priority:** HIGH
- **Effort:** 6-8 hours
- **Impact:** +3-5% MRR
- **Description:**
  - Generate embeddings for all graph entities
  - For queries: extract entities → find similar entities in graph
  - Use semantic similarity instead of exact name matching
  - Cache embeddings for performance
- **Success Criteria:**
  - >90% entity linking accuracy on test queries
  - <500ms query latency

---

## Phase 2: Semantic Relation Extraction (Est. +10-15% MRR)

### Task 2.1: LLM-Based Relation Extraction
- **Priority:** HIGH
- **Effort:** 8-10 hours
- **Impact:** +10-15% MRR
- **Description:**
  - Use Ollama (available locally) to extract relations
  - Create prompt template for relation extraction
  - Extract typed relations: parent_of, works_at, located_in, founded, etc.
  - Assign confidence scores based on LLM confidence
  - Rebuild graph with semantic relations
- **Success Criteria:**
  - Achieve >60% semantic relations (target: reduce co-occurrence from 99.6%)
  - Relation extraction precision ≥70%
  - Improved graph MRR ≥ BM25 baseline

### Task 2.2: Validate Relations Against Training Data
- **Priority:** MEDIUM
- **Effort:** 4-5 hours
- **Impact:** +2-3% MRR
- **Description:**
  - Sample 100 extracted relations
  - Manual validation by domain expert
  - Identify patterns in false positives
  - Adjust prompts to improve precision
- **Success Criteria:**
  - Identify precision bottlenecks
  - Prompt refinements documented

### Task 2.3: Fine-Tune Relation Weights
- **Priority:** MEDIUM
- **Effort:** 4-6 hours
- **Impact:** +3-5% MRR
- **Description:**
  - Weight relations by type (not all equal)
  - Learn weights from training set
  - Direct relations > indirect relations
  - High-confidence relations > low-confidence
- **Success Criteria:**
  - Graph traversal now outperforms baseline
  - Weights calibrated on dev set

---

## Phase 3: Learning-to-Rank (Est. +10-20% MRR)

### Task 3.1: Collect Relevance Judgments
- **Priority:** HIGH
- **Effort:** 8-12 hours (mostly manual)
- **Impact:** +10-20% MRR
- **Description:**
  - Annotate 50-100 query-document pairs
  - Rate relevance on scale: 0 (irrelevant) → 3 (perfect match)
  - Focus on queries where graph could help (cross-document, entity-centric)
  - Save annotations in standard format (TREC, JSON)
- **Success Criteria:**
  - 100+ annotated pairs
  - Inter-annotator agreement ≥80%

### Task 3.2: Engineer Retrieval Features
- **Priority:** HIGH
- **Effort:** 6-8 hours
- **Impact:** +5-10% MRR
- **Description:**
  - Extract features for each candidate:
    - BM25 score
    - Entity overlap count
    - Entity overlap confidence
    - Graph distance (shortest path)
    - Relation type match
    - Cross-document links
  - Normalize features to [0,1]
- **Success Criteria:**
  - 10+ features engineered
  - Feature correlation analysis done
  - No redundant features

### Task 3.3: Train Ranking Model
- **Priority:** MEDIUM
- **Effort:** 4-6 hours
- **Impact:** +5-15% MRR
- **Description:**
  - Simple LTR model (linear regression or LambdaMART)
  - 80/20 train/test split on annotated data
  - Optimize MRR on test set
  - Compare to hand-tuned hybrid weights
- **Success Criteria:**
  - Test MRR ≥ hand-tuned baseline
  - Model weights learned successfully

---

## Phase 4: Evaluation & Documentation

### Task 4.1: Comprehensive Evaluation
- **Priority:** HIGH
- **Effort:** 2-3 hours
- **Description:**
  - Run all methods on full test set
  - Per-topic performance breakdown
  - Identify topics where graph helps most
  - Compare all approaches
- **Success Criteria:**
  - Report showing final MRR for all methods
  - Topic breakdown analysis

### Task 4.2: Visualization & Analysis
- **Priority:** MEDIUM
- **Effort:** 3-4 hours
- **Description:**
  - Create performance comparison plots
  - Show improvement trajectory
  - Visualize example cross-document queries helped by graph
  - Write findings document
- **Success Criteria:**
  - Clear visualizations showing improvements
  - Document explaining findings

---

## Timeline & Milestones

```
Week 1: Phase 1 (Entity Extraction)
  - Mon-Wed: GLiNER integration + NER benchmark
  - Thu: spaCy integration
  - Fri: Entity embeddings

Week 2: Phase 2 (Relation Extraction)
  - Mon-Tue: LLM-based relation extraction
  - Wed: Graph rebuild + validation
  - Thu: Relation weight tuning
  - Fri: Graph performance measurement

Week 3: Phase 3 (Learning-to-Rank)
  - Mon-Tue: Collect relevance judgments
  - Wed: Feature engineering
  - Thu-Fri: Train ranking model

Week 4: Phase 4 (Evaluation)
  - Mon: Full evaluation
  - Tue-Wed: Visualization
  - Thu-Fri: Documentation & writeup
```

**Target:** +15-20% MRR improvement after 4 weeks

---

## Success Criteria (Overall)

| Metric | Baseline | Target | Status |
|--------|----------|--------|--------|
| Entity extraction precision | 65% | >85% | ⬜ |
| Semantic relations | 0.4% | >20% | ⬜ |
| Entity linking accuracy | 70% | >90% | ⬜ |
| Graph retrieval MRR | 0.05 | >0.68 | ⬜ |
| Simple hybrid MRR | 0.686 | >0.78 | ⬜ |
| Final LTR model MRR | N/A | >0.80 | ⬜ |

---

## Dependencies & Ordering

```
Phase 1: Entity Extraction
    ↓ (better entities for graph)
Phase 2: Relation Extraction
    ↓ (semantic relations for ranking)
Phase 3: Learning-to-Rank
    ↓ (optimal feature weighting)
Phase 4: Evaluation & Documentation
```

**Can run in parallel:** 
- Task 1.2 and 1.3 (different extraction methods)
- Task 2.1 and 2.2 (validation doesn't block)

---

## Resources Needed

- **Ollama** (for LLM relation extraction) - already available
- **spaCy** (optional, for better NER) - install if missing
- **GLiNER** (for high-quality NER) - deploy `scripts/gliner_server.py`
- **Human annotator** (for relevance judgments) - 8-12 hours
- **Compute** (for embeddings, ranking model) - modest requirements

---

## Files to Update/Create

| File | Task | Status |
|------|------|--------|
| `improved_extraction.py` | 1.1-1.3 | ✅ Exists |
| `gliner_ner.py` (NEW) | 1.1 | ⬜ Create |
| `entity_embeddings.py` (NEW) | 1.3 | ⬜ Create |
| `llm_relations.py` (NEW) | 2.1 | ⬜ Create |
| `relation_validator.py` (NEW) | 2.2 | ⬜ Create |
| `feature_engineer.py` (NEW) | 3.2 | ⬜ Create |
| `ltr_ranker.py` (NEW) | 3.3 | ⬜ Create |
| `run_final_eval.py` | 4.1 | ✅ Exists |
| `analysis.py` (NEW) | 4.2 | ⬜ Create |

---

## Notes

- Start with Phase 1.1 (GLiNER) - highest ROI
- Phase 2.1 (LLM relations) is next blocker - unlocks real improvement
- Phase 3 is optional but recommended - can yield +10-20%
- Document each phase completion for reference
- Test incrementally - measure each improvement

---

## Questions/Unknowns

- [ ] How many relevance judgments needed for good LTR model? (Est: 100+)
- [ ] What's the best LTR algorithm for this task? (Recommend: linear + RRF)
- [ ] Should we weight entities by frequency? (Probably yes)
- [ ] How to handle entity ambiguity? (Need disambiguation)

---

## Quick Start Commands

```bash
# Run baseline
python3 run_retrieval_experiment.py

# Run improved extraction
python3 build_improved_graph.py

# Run evaluation (after Phase 1.1)
python3 run_final_evaluation.py

# Check current performance
cat final_evaluation_results.json | jq '.results'
```

---

**Last Updated:** 2026-07-21  
**Owner:** [Assign to Agent/Team Member]  
**Status:** Ready to Start ✓
