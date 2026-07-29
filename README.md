# Dream RAG Research Project

A generalized model of dreaming in a RAG-based system. The project builds on the [kwaai-rag](https://github.com/kwaai/kwaai-rag) ingestion and knowledge-graph stack, with the goal of adding a "dream loop" that discovers cross-document links, completes entity schemas, and refines the graph during idle time.

**Implementation language:** Python. The `rust implementations/` directory contains reference ports from kwaai-rag and is not part of the active development path.

## Authors

- Christopher J. Mayfield
- Reza Rassool
- Jourdane Hamilton
- Annika Vriens
- Aman Avinash
- Maira Khwaja

## Publication

Paper in progress: *[How'd you sleep, bro? A Dreaming Retrieval-Augmented Generation Architecture Through the Lens of the Free Energy Principle](https://www.overleaf.com/read/gcrxgjcxqzqc#2e6eb7)* (Overleaf)

## Progress

### Done

The core kwaai-rag pipeline has been ported to Python. Each module lives in a top-level `.py` file.

| Module | Description |
|--------|-------------|
| `document.py` | Extract plain text from `.txt`, `.md`, `.pdf`, `.docx`, `.doc`, and other common formats |
| `chunker.py` | Text chunking — character-level sliding window and paragraph-semantic strategies |
| `doc_schema.py` | Document schema definitions, section matching, and auto-detection (YAML-driven) |
| `embedder.py` | Async HTTP client for the Ollama embedding API (`nomic-embed-text`, 768-dim) |
| `meta_store.py` | Per-tenant chunk metadata and file-sync tracking (SQLite) |
| `ner.py` | Lightweight proper-noun pre-screening and pronoun resolution (no external NLP deps) |
| `gliner.py` | Async HTTP client for a GLiNER NER server (multi-label + person hints) |
| `gliner_ner.py` | Sync multi-label GLiNER NER with post-filters, type floors, and OCR cleanup |
| `graph.py` | Knowledge graph with entity nodes, directed relations, LLM-based extraction, and SQLite persistence |
| `ingestion.py` | End-to-end pipeline: chunk → embed → upload, with optional knowledge-graph extraction |
| `corpus_schema.py` | Per-corpus entity/relation schema config (feeds the graph extraction types); sample in `schemas/` |
| `memory.py` | Memory-strength model: reinforcement + recency under a dynamically-modulated Ebbinghaus forgetting curve |
| `dream.py` | Dreaming consolidation loop — recomputes memory strengths, demotes weak chunks, synthesizes core facts |
| `dream_cycle.py` | Iterative sleep-like consolidation cycles over a graph DB |
| `improved_extraction.py` | GLiNER-first entity extraction with pattern/spaCy fallbacks and dedupe |
| `simulate_forgetting.py` | Simulate Ebbinghaus decay on a built graph and write retention timelines |

**Ingestion pipeline** (`ingestion.py`):

1. Extract text from a document
2. Split into chunks (configurable strategy, overlap, and surrounding context)
3. Embed chunks via Ollama
4. Store chunk metadata in `meta_store`
5. Optionally extract entities and relations into the knowledge graph (LLM + GLiNER + NER hints)

`graph.py` currently covers basic ingestion and extraction. The kwaai-rag reference (`rust implementations/graph.rs`) includes additional graph maintenance hooks that still need to be ported.

**Memory & dreaming** (`memory.py`, `dream.py`): every entity, relation, and chunk gets a *strength* combining how often it was reinforced (mention/evidence count) and how recently it was seen, decayed via an Ebbinghaus curve whose stability grows with reinforcement (the spacing effect). The dreaming loop reclassifies memory into `long_term` / `short_term` / `dormant`, demotes (never deletes) weak chunks, and optionally synthesizes consolidated "core facts" from the strongest entities. State is written to new tables (`memory_nodes`, `memory_edges`, `memory_chunks`, `consolidated_facts`) so ingestion is untouched. The query/structure/eval layers live in the sibling [`dreamrag-retrieval`](../dreamrag-retrieval) project.

```bash
python3 dream.py run                 # recompute strengths + consolidate
python3 dream.py run --synthesize    # also LLM-synthesize core facts
python3 dream.py show nodes          # strongest memories
python3 simulate_forgetting.py       # plot retention decay over time on a graph DB
```

### Recent work (graph retrieval + forgetting)

On the Corpus Final Review benchmark (~224 docs, 280 queries):

1. **GLiNER entity extraction** — `scripts/gliner_server.py`, `gliner_ner.py`, and `improved_extraction.py` replace noisy capitalization heuristics with filtered multi-label NER (person / org / location / work / …). Post-filters cut OCR junk and generic false positives (noisy entity names ~5.8% → ~0.1%).
2. **Graph build + hybrid retrieval** — `build_improved_graph.py` builds an entity graph; `simple_hybrid_retrieval.py` boosts BM25 with query-entity mentions; `improved_graph_retrieval.py` does multi-hop graph scoring.
3. **Evaluation** — `run_final_evaluation.py` compares BM25 vs Simple Hybrid vs Improved Graph. Latest results (`final_evaluation_results.json`):

| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR | vs BM25 |
|-----------|----------|----------|-----------|-----|---------|
| BM25 | 0.646 | 0.732 | 0.754 | **0.684** | baseline |
| Simple Hybrid | 0.646 | 0.750 | 0.761 | **0.692** | **+1.13%** |
| Improved Graph | 0.275 | 0.289 | 0.289 | 0.281 | −59% |

**Takeaway:** BM25 + entity boosting is the current win. Pure graph traversal is still behind because co-occurrence edges dominate and entity linking is exact-name only. Next levers: embedding-based linking and LLM semantic relations (see `TODO.md`, `IMPROVEMENTS_SUMMARY.md`).

4. **Ebbinghaus forgetting simulation** — `simulate_forgetting.py` applies `memory.py` retention \(R = e^{-t/S}\) to the improved graph (`mention_count` as reinforcement). With most entities mentioned once, average retention falls to ~0.24 by day 90 and ~97% of entities are dormant unless re-exposed. Output: `data/forgetting_simulation.json`.

```bash
# Optional: start GLiNER NER server
.venv/bin/python scripts/gliner_server.py --port 8000

# Rebuild cleaned entity graph
.venv/bin/python build_improved_graph.py --gliner-url http://127.0.0.1:8000

# Evaluate BM25 / hybrid / graph
.venv/bin/python run_final_evaluation.py

# Simulate forgetting over 365 days
.venv/bin/python simulate_forgetting.py
```

## Quick start

### 1. Install dependencies

```bash
cd dreamrag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Ollama and pull the embedding model

```bash
ollama pull nomic-embed-text
python -m dreamrag check
```

### 3. Add documents

Place files in `data/documents/` (`.txt`, `.md`, `.pdf`, `.docx`, etc.) or pass paths directly.

### 4. Ingest

```bash
# Chunk + embed + store metadata and vectors
python -m dreamrag ingest

# Also extract a knowledge graph (slower; uses your Ollama LLM)
python -m dreamrag ingest --graph --llm-model llama3.1:8b

# Ingest specific files
python -m dreamrag ingest path/to/file.pdf another.md
```

### 5. Check status

```bash
python -m dreamrag status
```

Ingested data is stored under `data/store/` (chunk metadata, vectors, and optional knowledge graph).

## TODO

### 1. Project wiring (run end-to-end)

- [x] Add `requirements.txt` with pinned dependencies
- [x] Add a minimal CLI (`python -m dreamrag ingest`)
- [x] Add `scripts/gliner_server.py` (GLiNER HTTP NER server)
- [x] Wire up a concrete vector store for chunk embeddings (`vector_store.py`)
- [ ] Organize modules into a proper Python package (e.g. move top-level modules into `dreamrag/`)

### 2. Dream loop (core research goal)

- [x] Implement `dream.py` — a consolidation runner operating over the graph during idle time
- [x] Memory-strength model (`memory.py`) — reinforcement + recency + Ebbinghaus forgetting
- [x] Consolidation — demote weak chunks to short-term/dormant, synthesize core facts from strong entities
- [x] Forgetting simulation — `simulate_forgetting.py` shows retention loss over time on the corpus graph
- [ ] Cross-link discovery — find entities shared across documents/chunks via `GraphStore.all_chunk_entity_pairs()`
- [ ] Relation completion — infer missing relations from graph structure and evidence chunks
- [ ] Entity schema completion — fill schema.org fields (`birthDate`, `addressLocality`, etc.) for low-confidence entities
- [ ] Post-dream graph refinement — dedup merges, relation sanitization, confidence rescoring
- [ ] Couple forgetting strengths into retrieval ranking (down-weight dormant facts)

### 3. Retrieval and query layer

Implemented in the sibling [`dreamrag-retrieval`](../dreamrag-retrieval) project, plus local graph-retrieval experiments in this repo:

- [x] Chunk vector search over embedded chunks
- [x] Hybrid retrieval combining vector search with graph neighbors
- [x] Structural router + query-likelihood reranking + cited answer generation (`ask`)
- [x] Structural understanding (clustering, summaries, timeline, relationship map)
- [x] Evaluation harness and pillar ablation study
- [x] Local BM25 / Simple Hybrid / Improved Graph eval on Corpus Final Review (`run_final_evaluation.py`)
- [x] GLiNER-backed entity extraction for graph build and query-time boosting
- [ ] Embedding-based entity linking (aliases / paraphrases)
- [ ] LLM semantic relation extraction (reduce co-occurrence-only edges)
- [ ] Learning-to-rank fusion of BM25 + graph features

### 4. Complete `graph.py`

`graph.py` covers basic ingestion but is missing most graph maintenance logic from the kwaai-rag reference. Port the following into Python:

- [ ] `search_entities` — entity retrieval by embedding
- [ ] `bfs_neighbors`, `entity_chunks` — graph traversal for RAG
- [ ] `find_dedup_candidates*` — entity deduplication (exact, fuzzy, name-structure, etc.)
- [ ] `merge_entity_into`, `unmerge_alias` — canonical entity merging
- [ ] `sanitize_relations` — clean up bad or inferred relations
- [ ] `coref_candidates_for_chunk` — coreference resolution
- [ ] `all_chunk_entity_pairs` — cross-link discovery for the dream loop
- [ ] `set_schema_type`, `set_document_titles` — dream completion helpers

### 5. Tests and examples

- [ ] Unit and integration tests for core modules (`pytest`)
- [x] Example document in `data/documents/sample.txt`
- [ ] Sample doc schemas (YAML) to exercise `doc_schema`
- [ ] CI pipeline (GitHub Actions)

### 6. Dreaming

- [ ] Assess how to optimize for dreaming, such as graph completion and storage compression
- [ ] Gather more information on the neuroscience
- [ ] Metrics for forgetting benefit (retrieval with vs without decay)


## Repository layout

```
document.py              # Text extraction
chunker.py               # Chunking strategies
doc_schema.py            # Section schemas
embedder.py              # Ollama embeddings
meta_store.py            # Chunk/sync metadata
vector_store.py          # Chunk embedding storage
ner.py                   # Proper-noun & pronoun handling
gliner.py                # Async GLiNER NER client
gliner_ner.py            # Sync GLiNER NER + quality filters
graph.py                 # Knowledge graph store & extraction
ingestion.py             # Full ingestion pipeline
corpus_schema.py         # Per-corpus entity/relation schema config
memory.py                # Memory-strength model (Ebbinghaus)
dream.py                 # Dreaming consolidation loop
dream_cycle.py           # Iterative dream cycles over a graph DB
simulate_forgetting.py   # Forgetting-curve simulation CLI
improved_extraction.py   # GLiNER-first entity extraction
build_improved_graph.py  # Build cleaned entity graph from corpus
simple_hybrid_retrieval.py
improved_graph_retrieval.py
run_final_evaluation.py  # BM25 vs hybrid vs graph eval
scripts/gliner_server.py # GLiNER HTTP server
schemas/                 # Sample per-corpus schemas
dreamrag/                # CLI (`python -m dreamrag`)
data/documents/          # Drop files here for ingestion
data/store/              # Generated databases (gitignored)
data/forgetting_simulation.json

rust implementations/    # kwaai-rag reference (not actively maintained)
  *.rs
```

## Dependencies

**Python packages** — `aiohttp` (embedder, graph, gliner, ingestion), `pdfminer.six` / `pdfplumber` (PDF extraction), `pyyaml` (doc schemas). Optional for NER experiments: `gliner`, `flask`, `rank_bm25`, `openpyxl`.

**External services** — Ollama (embeddings), an LLM inference endpoint (graph extraction), and optionally a GLiNER NER server (`scripts/gliner_server.py`).

## Ideas

- Evaluate recall (and MRR), not just pulled context
- Wire forgetting strengths into live retrieval ranking
- Semantic relations + entity linking before more graph traversal complexity
