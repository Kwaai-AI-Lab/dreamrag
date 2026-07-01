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
| `gliner.py` | Thin async client for a GLiNER NER server — injects high-confidence person spans into extraction prompts |
| `graph.py` | Knowledge graph with entity nodes, directed relations, LLM-based extraction, and SQLite persistence |
| `ingestion.py` | End-to-end pipeline: chunk → embed → upload, with optional knowledge-graph extraction |
| `corpus_schema.py` | Per-corpus entity/relation schema config (feeds the graph extraction types); sample in `schemas/` |
| `memory.py` | Memory-strength model: reinforcement + recency under a dynamically-modulated Ebbinghaus forgetting curve |
| `dream.py` | Dreaming consolidation loop — recomputes memory strengths, demotes weak chunks, synthesizes core facts |

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
- [ ] Add `scripts/gliner_server.py` (referenced by `gliner.py` but not yet in this repo)
- [x] Wire up a concrete vector store for chunk embeddings (`vector_store.py`)
- [ ] Organize modules into a proper Python package (e.g. move top-level modules into `dreamrag/`)

### 2. Dream loop (core research goal)

- [x] Implement `dream.py` — a consolidation runner operating over the graph during idle time
- [x] Memory-strength model (`memory.py`) — reinforcement + recency + Ebbinghaus forgetting
- [x] Consolidation — demote weak chunks to short-term/dormant, synthesize core facts from strong entities
- [ ] Cross-link discovery — find entities shared across documents/chunks via `GraphStore.all_chunk_entity_pairs()`
- [ ] Relation completion — infer missing relations from graph structure and evidence chunks
- [ ] Entity schema completion — fill schema.org fields (`birthDate`, `addressLocality`, etc.) for low-confidence entities
- [ ] Post-dream graph refinement — dedup merges, relation sanitization, confidence rescoring

### 3. Retrieval and query layer

Implemented in the sibling [`dreamrag-retrieval`](../dreamrag-retrieval) project:

- [x] Chunk vector search over embedded chunks
- [x] Hybrid retrieval combining vector search with graph neighbors
- [x] Structural router + query-likelihood reranking + cited answer generation (`ask`)
- [x] Structural understanding (clustering, summaries, timeline, relationship map)
- [x] Evaluation harness and pillar ablation study

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
- [ ] Metrics


## Repository layout

```
document.py       # Text extraction
chunker.py        # Chunking strategies
doc_schema.py     # Section schemas
embedder.py       # Ollama embeddings
meta_store.py     # Chunk/sync metadata
vector_store.py   # Chunk embedding storage
ner.py            # Proper-noun & pronoun handling
gliner.py         # GLiNER NER client
graph.py          # Knowledge graph store & extraction
ingestion.py      # Full ingestion pipeline
corpus_schema.py  # Per-corpus entity/relation schema config
memory.py         # Memory-strength model (reinforcement + recency + Ebbinghaus)
dream.py          # Dreaming consolidation loop (`python3 dream.py run`)
schemas/          # Sample per-corpus schemas (e.g. manhattan_project.json)
dreamrag/         # CLI (`python -m dreamrag`)
data/documents/   # Drop files here for ingestion
data/store/       # Generated databases (gitignored)

rust implementations/   # kwaai-rag reference (not actively maintained)
  *.rs
```

## Dependencies

**Python packages** — `aiohttp` (embedder, graph, gliner, ingestion), `pdfminer.six` (PDF extraction), `pyyaml` (doc schemas).

**External services** — Ollama (embeddings), an LLM inference endpoint (graph extraction), and optionally a GLiNER NER server.


## Ideas 
 -evaluate recall and not just pulled context 
 