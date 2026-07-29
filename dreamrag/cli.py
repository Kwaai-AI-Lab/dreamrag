"""
dreamrag.cli — Command-line interface for document ingestion.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chunker import ChunkConfig, ChunkStrategy
from document import SUPPORTED_EXTENSIONS, extract_text
from embedder import DEFAULT_DIM, DEFAULT_MODEL, EmbedClient
from graph import GraphStore
from ingestion import GraphIngestConfig, IngestConfig, ingest_text
from meta_store import MetaStore, SyncMeta
from vector_store import VectorStore

DEFAULT_TENANT_ID = UUID("00000000-0000-4000-8000-000000000001")
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_DOCS_DIR = DEFAULT_DATA_DIR / "documents"
DEFAULT_STORE_DIR = DEFAULT_DATA_DIR / "store"

SUPPORTED_GLOBS = [f"*.{ext}" for ext in SUPPORTED_EXTENSIONS]
SKIP_FILENAMES = {"README.md", "readme.md"}


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def _collect_files(paths: list[Path], docs_dir: Path) -> list[Path]:
    if not paths:
        if not docs_dir.exists():
            docs_dir.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []
        for pattern in SUPPORTED_GLOBS:
            files.extend(docs_dir.glob(pattern))
        return sorted(p for p in set(files) if p.name not in SKIP_FILENAMES)

    out: list[Path] = []
    for path in paths:
        path = path.resolve()
        if path.is_dir():
            for pattern in SUPPORTED_GLOBS:
                out.extend(path.glob(pattern))
        elif path.is_file():
            out.append(path)
        else:
            raise FileNotFoundError(f"not found: {path}")
    return sorted(set(out))


def _doc_name(path: Path, docs_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(docs_dir.resolve()))
    except ValueError:
        return path.name


def _needs_reingest(meta: MetaStore, doc_name: str, path: Path) -> bool:
    sync = meta.get_sync_meta(doc_name)
    if sync is None:
        return True
    stat = path.stat()
    return sync.mtime_secs != int(stat.st_mtime) or sync.file_size != stat.st_size


async def _check_ollama(embed: EmbedClient) -> int:
    dim = await embed.probe_dim()
    print(f"Ollama OK — embed model '{embed.model}' returns {dim}-dim vectors")
    return dim


async def _cmd_check(args: argparse.Namespace) -> int:
    embed = EmbedClient(base_url=args.ollama_url, model=args.embed_model)
    try:
        await _check_ollama(embed)
    except Exception as e:
        print(f"Ollama check failed: {e}", file=sys.stderr)
        if "not found" in str(e).lower():
            print(f"  Try: ollama pull {args.embed_model}", file=sys.stderr)
        return 1
    finally:
        await embed.close()
    return 0


def _open_stores(data_dir: Path, dim: int) -> tuple[MetaStore, VectorStore, GraphStore]:
    store_dir = data_dir / "store"
    store_dir.mkdir(parents=True, exist_ok=True)
    meta = MetaStore.open(store_dir, DEFAULT_TENANT_ID)
    vectors = VectorStore.open(store_dir, DEFAULT_TENANT_ID, dim)
    graph = GraphStore.open(store_dir, DEFAULT_TENANT_ID)
    return meta, vectors, graph


async def _ingest_file(
    path: Path,
    docs_dir: Path,
    meta: MetaStore,
    vectors: VectorStore,
    embed: EmbedClient,
    cfg: IngestConfig,
    force: bool,
) -> bool:
    doc_name = _doc_name(path, docs_dir)
    if not force and not _needs_reingest(meta, doc_name, path):
        print(f"  skip (unchanged): {doc_name}")
        return False

    print(f"  ingest: {doc_name}")
    text = extract_text(path)
    if not text.strip():
        print(f"  warning: no text extracted from {doc_name}", file=sys.stderr)
        return False

    old_ids = meta.delete_doc(doc_name)
    if old_ids:
        vectors.delete(old_ids)

    async def upload_fn(batch: list[tuple[int, list[float]]]) -> int:
        return vectors.upsert_batch(batch)

    def progress(done: int, total: int) -> None:
        print(f"    embedded {done}/{total} chunks", end="\r")

    result = await ingest_text(cfg, meta, doc_name, text, upload_fn, progress)
    print(
        f"    done: {result.chunks_ingested} chunks, "
        f"{result.vectors_uploaded} vectors uploaded"
    )

    stat = path.stat()
    meta.put_sync_meta(
        doc_name,
        SyncMeta(
            file_path=str(path),
            mtime_secs=int(stat.st_mtime),
            file_size=stat.st_size,
        ),
    )
    return True


async def _cmd_ingest(args: argparse.Namespace) -> int:
    docs_dir = Path(args.docs_dir)
    data_dir = Path(args.data_dir)
    files = _collect_files([Path(p) for p in args.paths], docs_dir)

    if not files:
        print(f"No documents found. Add files to {docs_dir} or pass paths explicitly.")
        print(f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
        return 1

    embed = EmbedClient(base_url=args.ollama_url, model=args.embed_model)
    try:
        dim = await _check_ollama(embed)
    except Exception as e:
        print(f"Cannot connect to Ollama: {e}", file=sys.stderr)
        print(f"  Start Ollama, then run: ollama pull {args.embed_model}", file=sys.stderr)
        await embed.close()
        return 1

    meta, vectors, graph = _open_stores(data_dir, dim)

    chunk_cfg = ChunkConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        strategy=ChunkStrategy.Paragraph if args.paragraph_chunks else ChunkStrategy.Character,
    )

    graph_cfg = None
    if args.graph:
        graph_cfg = GraphIngestConfig(
            store=graph,
            inference_url=args.ollama_url,
            model=args.llm_model,
            workers=args.workers,
            context_window=args.context_window,
        )
        if args.gliner_url:
            from gliner import GliNERClient
            graph_cfg.gliner_client = GliNERClient(args.gliner_url)

    cfg = IngestConfig(embed=embed, chunk_cfg=chunk_cfg, graph=graph_cfg)

    print(f"Ingesting {len(files)} file(s) into {data_dir / 'store'}")
    ingested = 0
    vector_count = 0
    entity_count = 0
    relation_count = 0
    try:
        for path in files:
            if await _ingest_file(path, docs_dir, meta, vectors, embed, cfg, args.force):
                ingested += 1
        vector_count = vectors.count()
        entity_count = graph.node_count()
        relation_count = graph.relation_count()
    finally:
        await embed.close()
        meta.close()
        vectors.close()
        graph.close()

    print(f"Finished — {ingested} document(s) ingested, {vector_count} vectors total")
    if args.graph:
        print(f"Knowledge graph: {entity_count} entities, {relation_count} relations")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    store_dir = data_dir / "store"
    if not store_dir.exists():
        print(f"No data store yet. Run: python -m dreamrag ingest")
        return 0

    meta = MetaStore.open(store_dir, DEFAULT_TENANT_ID)
    vectors = VectorStore.open_existing(store_dir, DEFAULT_TENANT_ID, DEFAULT_DIM)
    graph = GraphStore.open(store_dir, DEFAULT_TENANT_ID)

    docs = meta.list_docs()
    chunks = meta.all_chunks()
    print(f"Data directory: {data_dir}")
    print(f"Documents:      {len(docs)}")
    print(f"Chunks:         {len(chunks)}")
    print(f"Vectors:        {vectors.count()}")
    print(f"Graph entities: {graph.node_count()}")
    print(f"Graph relations: {graph.relation_count()}")
    if docs:
        print("\nDocuments:")
        for name in docs:
            sync = meta.get_sync_meta(name)
            status = "synced" if sync else "no sync metadata"
            print(f"  - {name} ({status})")

    meta.close()
    vectors.close()
    graph.close()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dreamrag",
        description="Dream RAG — ingest, eval, and forgetting simulation",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Verify Ollama and the embedding model")
    check.add_argument("--ollama-url", default="http://localhost:11434")
    check.add_argument("--embed-model", default=DEFAULT_MODEL)

    ingest = sub.add_parser("ingest", help="Ingest document(s)")
    ingest.add_argument(
        "paths",
        nargs="*",
        help="Files or directories to ingest (default: all files in data/documents/)",
    )
    ingest.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ingest.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR))
    ingest.add_argument("--ollama-url", default="http://localhost:11434")
    ingest.add_argument("--embed-model", default=DEFAULT_MODEL)
    ingest.add_argument("--llm-model", default="llama3.1:8b")
    ingest.add_argument("--chunk-size", type=int, default=800)
    ingest.add_argument("--chunk-overlap", type=int, default=200)
    ingest.add_argument("--paragraph-chunks", action="store_true")
    ingest.add_argument("--graph", action="store_true", help="Extract knowledge graph entities/relations")
    ingest.add_argument("--gliner-url", default="", help="Optional GLiNER NER server URL")
    ingest.add_argument("--workers", type=int, default=1)
    ingest.add_argument("--context-window", type=int, default=1)
    ingest.add_argument("--force", action="store_true", help="Re-ingest even if file unchanged")

    status = sub.add_parser("status", help="Show ingestion status")
    status.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))

    forget = sub.add_parser(
        "forget",
        help="Simulate Ebbinghaus forgetting on a knowledge graph",
    )
    forget.add_argument(
        "--graph-db",
        default=str(DEFAULT_STORE_DIR / "corpus_graph_improved.db"),
        help="Path to graph SQLite DB",
    )
    forget.add_argument(
        "--out",
        default=str(DEFAULT_DATA_DIR / "forgetting_simulation.json"),
        help="Output JSON path",
    )
    forget.add_argument("--max-days", type=int, default=365)
    forget.add_argument("--step", type=int, default=5)
    forget.add_argument("--base-stability", type=float, default=30.0)
    forget.add_argument("--boost", type=float, default=1.5)

    evaluate = sub.add_parser(
        "eval",
        help="Local retrieval eval: BM25 vs Simple Hybrid vs Improved Graph",
    )
    evaluate.add_argument(
        "--corpus-path",
        default="/Users/christophermayfield/Desktop/Corpus_Final_Review",
        help="Path to Corpus_Final_Review (or compatible corpus)",
    )
    evaluate.add_argument(
        "--graph-db",
        default=str(DEFAULT_STORE_DIR / "corpus_graph_improved.db"),
        help="Improved graph SQLite DB",
    )
    evaluate.add_argument(
        "--out",
        default=str(ROOT / "final_evaluation_results.json"),
        help="Output JSON path",
    )
    evaluate.add_argument(
        "--gliner-url",
        default="http://127.0.0.1:8000",
        help="GLiNER NER server URL",
    )
    evaluate.add_argument(
        "--skip-graph",
        action="store_true",
        help="Only run BM25 and Simple Hybrid",
    )

    return parser


def _cmd_forget(args: argparse.Namespace) -> int:
    from simulate_forgetting import run_simulation

    return run_simulation(
        graph_db=Path(args.graph_db),
        out=Path(args.out),
        max_days=args.max_days,
        step=args.step,
        base_stability=args.base_stability,
        boost=args.boost,
    )


def _cmd_eval(args: argparse.Namespace) -> int:
    from run_final_evaluation import run_evaluation

    return run_evaluation(
        corpus_path=Path(args.corpus_path),
        graph_db=Path(args.graph_db),
        out_path=Path(args.out),
        gliner_url=args.gliner_url or None,
        skip_graph=args.skip_graph,
    )


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if args.command == "check":
        code = asyncio.run(_cmd_check(args))
    elif args.command == "ingest":
        code = asyncio.run(_cmd_ingest(args))
    elif args.command == "status":
        code = _cmd_status(args)
    elif args.command == "forget":
        code = _cmd_forget(args)
    elif args.command == "eval":
        code = _cmd_eval(args)
    else:
        parser.error(f"unknown command: {args.command}")
        return

    raise SystemExit(code)


if __name__ == "__main__":
    main()
