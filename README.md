# Dream RAG Research Project

A generalized model of dreaming in a RAG-based system. The project builds on the [kwaai-rag](https://github.com/kwaai/kwaai-rag) ingestion and knowledge-graph stack, with the goal of adding a "dream loop" that discovers cross-document links, completes entity schemas, and refines the graph during idle time.

## Authors

- Christopher J. Mayfield
- Reza Rassool
- Jourdane Hamilton
- Annika Vriens
- Aman Avinash
- Maira Khwaja

## Progress

### Done

The core kwaai-rag pipeline has been brought into this repo in both **Rust** (reference implementation) and **Python** (research/prototyping ports). Each Python module mirrors its Rust counterpart.

| Module | Description |
|--------|-------------|
| `document` | Extract plain text from `.txt`, `.md`, `.pdf`, `.docx`, `.doc`, and other common formats |
| `chunker` | Text chunking — character-level sliding window and paragraph-semantic strategies |
| `doc_schema` | Document schema definitions, section matching, and auto-detection (YAML-driven) |
| `embedder` | Async HTTP client for the Ollama embedding API (`nomic-embed-text`, 768-dim) |
| `meta_store` | Per-tenant chunk metadata and file-sync tracking (SQLite) |
| `ner` | Lightweight proper-noun pre-screening and pronoun resolution (no external NLP deps) |
| `gliner` | Thin async client for a GLiNER NER server — injects high-confidence person spans into extraction prompts |
| `graph` | Knowledge graph with entity nodes, directed relations, LLM-based extraction, and SQLite persistence |
| `ingestion` | End-to-end pipeline: chunk → embed → upload, with optional knowledge-graph extraction |

**Ingestion pipeline** (from `ingestion`):

1. Extract text from a document
2. Split into chunks (configurable strategy, overlap, and surrounding context)
3. Embed chunks via Ollama
4. Store chunk metadata in `meta_store`
5. Optionally extract entities and relations into the knowledge graph (LLM + GLiNER + NER hints)

The Rust `graph` module already includes hooks and data structures intended for the dream loop (e.g. `evidence` chunk tracking, cross-link discovery via `all_chunk_entity_pairs`, schema.org type completion). These are not yet wired into a standalone dream task runner in this repo.

### Not yet started

- Dream loop implementation (cross-link discovery, relation completion, graph refinement during idle time)
- Retrieval and query interface
- Integration tests and example scripts
- Package manifests (`Cargo.toml`, `requirements.txt`) and project wiring

## Repository layout

```
document.{rs,py}     # Text extraction
chunker.{rs,py}      # Chunking strategies
doc_schema.{rs,py}  # Section schemas
embedder.{rs,py}     # Ollama embeddings
meta_store.{rs,py}  # Chunk/sync metadata
ner.{rs,py}          # Proper-noun & pronoun handling
gliner.{rs,py}       # GLiNER NER client
graph.{rs,py}        # Knowledge graph store & extraction
ingestion.{rs,py}   # Full ingestion pipeline
```

## Dependencies (by module)

**Python** — `aiohttp` (embedder, graph, gliner, ingestion), `pdfminer.six` (PDF extraction), `pyyaml` (doc schemas).

**Rust** — modules are self-contained source files ported from kwaai-rag; a `Cargo.toml` has not been added yet.

**External services** — Ollama (embeddings), an LLM inference endpoint (graph extraction), and optionally a GLiNER NER server.
