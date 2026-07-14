# DreamRAG Forgetting Curve Experiment - Complete Roadmap

## What We Have ✅

1. **Complete Corpus** — 224 documents indexed (15 topics, 280 QA pairs)
2. **BM25 Baseline** — MRR 0.684 (strong lexical retrieval)
3. **Forgetting Curve Model** — `memory.py` (Ebbinghaus-based strength calculation)
4. **Evaluation Framework** — Recall@K, MRR, nDCG metrics
5. **Test Cases** — 280 questions with ground truth from QA-Trackers

## What We Need (Prioritized)

### Phase 1: Graph Construction & Population (High Priority)

**1.1 Entity & Relation Extraction**
- Extract named entities from corpus documents (people, places, concepts, dates)
- Extract relationships between entities ("X wrote Y", "Y is in Z", etc.)
- Tools: Could use:
  - GLiNER (lightweight, no training needed)
  - spaCy NER + custom relation extraction
  - Rule-based extraction for structured data (legal cases, RFC numbers)
- Output: List of (entity1, relation, entity2) triples

**1.2 Knowledge Graph Construction**
- Build graph data structure:
  - Nodes: Entities with attributes (type, source documents, confidence)
  - Edges: Relations with attributes (type, source documents, confidence)
- Storage: Could use:
  - In-memory Python dict (simple, fast for evaluation)
  - NetworkX library (graph algorithms built-in)
  - Neo4j (persistent, queryable, but more heavyweight)
- Implement: `class KnowledgeGraph` with add_node, add_edge, get_neighbors methods

**1.3 Initial Strength Assignment**
- Every node/edge starts with strength 1.0 and last_seen = creation_time
- Reinforce ment counter = 1 (first occurrence)
- State classification: Most will be SHORT_TERM or LONG_TERM initially

### Phase 2: Memory Strength Integration (High Priority)

**2.1 Implement Strength Tracking**
```python
class GraphNode:
    entity: str
    type: str
    source_documents: List[str]
    reinforcement: int          # Times this entity was validated
    last_seen: datetime         # When it was last encountered
    strength: float             # Current memory strength (0-1)
    state: str                  # "long_term", "short_term", "dormant"
    
    def update_strength(self, now: datetime, params: MemoryParams):
        elapsed = elapsed_days(self.last_seen, now)
        self.strength = strength(self.reinforcement, elapsed, params)
        self.state = classify(self.strength, params)
```

**2.2 Refresh Mechanism**
- When a fact is retrieved/validated, increment reinforcement += 1
- Update last_seen = current_time
- Recalculate strength
- Expected: Frequently used facts become more stable

### Phase 3: Dream Cycle Implementation (High Priority)

**3.1 Periodic Refinement Loop**
```python
def dream_cycle(graph, params, now):
    """One iteration of graph refinement."""
    # 1. Update strength for all nodes/edges
    for node in graph.nodes:
        node.update_strength(now, params)
    
    # 2. Classify states
    for node in graph.nodes:
        if node.state == "dormant" and node.strength < 0.2:
            mark_for_pruning(node)
    
    # 3. Prune or archive weak facts
    prune_dormant_nodes(graph)
    
    # 4. Deduplication/consolidation (merge similar entities)
    consolidate_duplicates(graph)
    
    # 5. Return metrics
    return {
        'nodes_pruned': count_pruned,
        'nodes_consolidated': count_consolidated,
        'avg_strength': avg_node_strength(graph),
        'entity_coverage': len(graph.nodes),
    }
```

**3.2 Schedule Dream Cycles**
- After every N queries? (e.g., every 50 questions)
- After every M days of simulated time? (e.g., every 30 days)
- Measure: How does graph evolve? Does it stabilize?

### Phase 4: Graph-Based Retrieval (High Priority)

**4.1 Entity-Aware Query Processing**
```python
def retrieve_with_graph(query, graph, bm25, params):
    """
    1. Extract entities from query
    2. Find matching nodes in graph
    3. Traverse neighbors (cross-document retrieval)
    4. Score by entity strength + document relevance
    """
    # Extract entities from query
    query_entities = extract_entities(query)
    
    # Find matching graph nodes
    relevant_nodes = [n for n in graph.nodes if n.entity in query_entities]
    
    # Get documents mentioning these entities
    relevant_docs = set()
    for node in relevant_nodes:
        relevant_docs.update(node.source_documents)
    
    # Also get neighbors (cross-document retrieval)
    for node in relevant_nodes:
        for neighbor in graph.get_neighbors(node):
            relevant_docs.update(neighbor.source_documents)
    
    # Rank documents by entity strength + BM25 score
    return rank_by_strength_and_relevance(relevant_docs, relevant_nodes, bm25)
```

**4.2 Cross-Document Query Answering**
- Query: "How do X and Y relate?"
- Find nodes for X and Y in graph
- Traverse path between them
- Return path + supporting documents

### Phase 5: Reranking & Hybrid Approaches (Medium Priority)

**5.1 Cross-Encoder Reranking**
- Fine-tune or use pre-trained cross-encoder on QA-pairs
- Input: (query, document)
- Output: Relevance score (0-1)
- Rerank BM25 results
- Estimated gain: +5-10% MRR

**5.2 Hybrid Fusion (RRF)**
- Combine multiple rankers:
  - BM25 lexical search
  - Graph-based entity retrieval
  - (Optional) Dense embeddings
- Reciprocal Rank Fusion formula: score = Σ 1/(k + rank)
- Expected gain: +10-15% MRR

**5.3 Dense Embeddings (Optional)**
- Use sentence-transformers to embed documents
- At query time: embed query, find nearest neighbors
- Combine with BM25 via RRF
- Expected gain: +10-20% MRR on semantic queries

### Phase 6: Evaluation & Metrics (High Priority)

**6.1 Retrieval Metrics (Already Have)**
- Recall@K ✓
- MRR ✓
- nDCG@K ✓

**6.2 Graph Quality Metrics (Need to Add)**
- **Entity Coverage** — How many ground-truth entities are in the graph?
- **Relation Completeness** — How many ground-truth relations are captured?
- **Compression Delta** — How much did graph shrink during pruning? (efficiency)
- **Deduplication Delta** — How many duplicate entities were merged?
- **Strength Distribution** — What % of nodes are in each state (long_term, short_term, dormant)?

**6.3 Retrieval + Forgetting Metrics (New)**
- **Fact Confidence** — Average strength of entities in retrieved documents
- **Stability vs Rank** — Do high-strength entities rank higher?
- **Forgetting Impact** — How much does pruning hurt retrieval? (should be minimal)
- **Reinforcement Effect** — For repeated queries, does strength increase improve ranking?

**6.4 Experimental Design**
```python
# Hypothesis: DreamRAG (with forgetting curve) outperforms BM25
# Control: BM25 baseline
# Treatment: DreamRAG with graph + forgetting + dream cycles

results = {
    'baseline_bm25': {
        'recall@1': 0.646,
        'recall@10': 0.754,
        'mrr': 0.684,
    },
    'dreamrag_no_refinement': {
        'recall@1': ?,
        'recall@10': ?,
        'mrr': ?,
    },
    'dreamrag_with_refinement': {
        'recall@1': ?,
        'recall@10': ?,
        'mrr': ?,
        'graph_metrics': { ... }
    }
}
```

### Phase 7: Visualization & Reporting (Medium Priority)

**7.1 Graph Visualizations**
- Show graph structure over time (node/edge counts, entity types)
- Heatmap: Entity strength distribution before/after refinement
- Timeline: How graph evolves over dream cycles

**7.2 Comparative Dashboards**
- BM25 vs DreamRAG (graph-based) vs Hybrid retrieval
- Per-topic breakdowns
- Query difficulty analysis (easy vs hard questions)

**7.3 Detailed Analysis Report**
- Which questions did DreamRAG help? Which didn't it help?
- Which entities were pruned? Did we lose coverage?
- Computational cost: Time/memory for dream cycles vs retrieval

---

## Implementation Order (Recommended)

### Week 1: Core Graph + Strength Tracking
1. Implement entity/relation extraction
2. Build knowledge graph structure
3. Integrate `memory.py` strength calculations
4. Test on small corpus sample

### Week 2: Dream Cycle + Graph Retrieval
5. Implement dream cycle (strength update + pruning)
6. Implement graph-based retrieval (entity matching, neighbor traversal)
7. Measure graph metrics (entity coverage, etc.)

### Week 3: Evaluation & Comparison
8. Run full evaluation: BM25 vs DreamRAG (no refinement) vs DreamRAG (with refinement)
9. Compute metrics at each stage
10. Create comparative dashboards

### Week 4: Optimization & Analysis
11. Implement reranking/hybrid fusion (optional, if time)
12. Analyze which questions benefited from graph
13. Write up findings

---

## File Structure (To Create)

```
dreamrag/
├── graph.py                    # (exists) Knowledge graph structure
├── memory.py                   # (exists) Forgetting curve model
├── ner.py                      # (exists) Entity extraction
├── graph_refinement.py         # (NEW) Dream cycle implementation
├── graph_retrieval.py          # (NEW) Entity-based retrieval
├── experiment.py               # (NEW) Full experimental pipeline
└── eval/
    ├── graph_metrics.py        # (exists) Graph quality metrics
    ├── retrieval_metrics.py    # (exists) Retrieval metrics
    └── experiment_runner.py    # (NEW) Run full experiment & compare
```

---

## Critical Questions to Answer

1. **Entity Extraction Quality** — How accurate is NER on this diverse corpus?
   - Ground truth: Manually annotate 50 documents → measure precision/recall

2. **Graph Size** — Will the graph grow unbounded?
   - Estimate: 224 docs × avg 20 entities/doc = ~4,500 nodes
   - With dedup/consolidation: Maybe 2,000-3,000 stable nodes

3. **Dream Cycle Efficiency** — How often should we run it?
   - Too often: Expensive, thrashes graph
   - Too rare: Doesn't consolidate weak facts
   - Hypothesis: Every 50-100 queries is sweet spot

4. **Forgetting Curve Validation** — Do the parameters (30-day stability, 0.66/0.33 thresholds) work?
   - Ground truth: Manual evaluation of which facts are "truly core"
   - Measure: Does strength correlate with human judgement?

5. **Retrieval Improvement** — Will graph help?
   - BM25 already at 0.684 MRR (good)
   - Graph might help with:
     - Cross-document queries ("How do X and Y relate?")
     - Synonym/paraphrase matching (via entity linking)
   - But might hurt on:
     - Very short queries (less entity signal)
     - Questions already well-answered by lexical search

---

## Success Criteria

| Goal | Target | Why |
|------|--------|-----|
| Entity coverage | ≥80% of ground-truth entities extracted | Ensures graph represents corpus well |
| Graph stability | Converges after 5-10 dream cycles | Forgetting curve prevents unbounded growth |
| Retrieval with graph | MRR ≥0.65 (not worse than BM25) | Ensure graph doesn't harm retrieval |
| Forgetting benefit | Pruned <20% of nodes, 0% hurt retrieval | Selective forgetting improves efficiency without hurting quality |
| Cross-document QA | 10%+ improvement on multi-hop questions | Graph should help on complex reasoning |
| Performance | <500ms/query with graph (vs ~50ms BM25) | Acceptable latency tradeoff |

---

## Open Questions

1. Should we implement full duplicate entity consolidation (expensive) or simple merging (fast)?
2. How to handle entity ambiguity? (E.g., "Python" = language vs snake vs company)
3. Should reinforcement come from:
   - Manual validation only (clean but limited signal)
   - Ranking signals (implicit feedback when user picks that result)
   - Both?
4. How to measure "cross-document" question answering if QA-Trackers only have single-document ground truth?

---

## Rough Timeline

- **Now → 3 weeks**: Implement core + dream cycle
- **Weeks 3-4**: Full evaluation & analysis
- **Week 4+**: Publication-ready visualizations & report

This is ambitious but doable. The critical path is: **entity extraction → graph construction → dream cycle → evaluation**.

Once you have dream cycles working, the forgetting curve model will shine—you'll see the graph stabilize, weak facts fade, and retrieval potentially improve on complex questions.

**Ready to start with entity extraction?**
