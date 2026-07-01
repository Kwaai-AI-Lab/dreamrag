"""
dream.py — The dreaming consolidation loop.

Runs unsupervised over the knowledge graph during idle time and:
  1. recomputes memory strength for every entity, relation, and chunk
     (recency + reinforcement via the Ebbinghaus model in memory.py);
  2. classifies each into long_term / short_term / dormant;
  3. *consolidates* — optionally synthesizes natural-language "core facts" from
     the strongest entities and their relations, persisting them to the graph;
  4. *demotes* weak raw chunks to short-term/dormant (kept, never deleted) so
     retrieval can prioritize consolidated, high-strength memory.

Memory state is written to new tables in the graph DB so the existing ingestion
pipeline is untouched:
    memory_nodes, memory_edges, memory_chunks, consolidated_facts

Usage:
    python3 dream.py run                 # recompute strengths + demote chunks
    python3 dream.py run --synthesize    # also LLM-synthesize core facts
    python3 dream.py show nodes|chunks|facts
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from memory import (
    DORMANT,
    LONG_TERM,
    SHORT_TERM,
    MemoryParams,
    classify,
    elapsed_days,
    parse_timestamp,
    stability,
    strength,
)

DEFAULT_TENANT_ID = UUID("00000000-0000-4000-8000-000000000001")
DEFAULT_STORE_DIR = Path(__file__).resolve().parent / "data" / "store"


# ── small synchronous Ollama client (only used for --synthesize) ───────────────
def _ollama_chat(prompt: str, model: str, base_url: str, num_predict: int = 120) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": num_predict},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("message", {}).get("content", "").strip()


class DreamLoop:
    def __init__(
        self,
        store_dir: Path = DEFAULT_STORE_DIR,
        tenant_id: UUID = DEFAULT_TENANT_ID,
        params: MemoryParams | None = None,
        now: datetime | None = None,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.tenant_id = tenant_id
        self.params = params or MemoryParams()
        self.now = now or datetime.now(timezone.utc)
        self.graph_path = self.store_dir / f"graph-{tenant_id}.db"
        self.meta_path = self.store_dir / f"{tenant_id}.db"
        if not self.graph_path.exists():
            raise FileNotFoundError(f"graph store not found: {self.graph_path}")
        self._gcon = sqlite3.connect(str(self.graph_path))
        self._gcon.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        self._gcon.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_nodes (
                entity_id INTEGER PRIMARY KEY, name TEXT, entity_type TEXT,
                reinforcement INTEGER, last_seen TEXT, stability REAL,
                strength REAL, state TEXT
            );
            CREATE TABLE IF NOT EXISTS memory_edges (
                src_id INTEGER, dst_id INTEGER, relation_type TEXT,
                reinforcement INTEGER, last_seen TEXT, strength REAL, state TEXT,
                PRIMARY KEY (src_id, dst_id, relation_type)
            );
            CREATE TABLE IF NOT EXISTS memory_chunks (
                chunk_id INTEGER PRIMARY KEY, reinforcement INTEGER,
                last_seen TEXT, strength REAL, state TEXT
            );
            CREATE TABLE IF NOT EXISTS consolidated_facts (
                entity_id INTEGER PRIMARY KEY, fact TEXT, strength REAL
            );
            """
        )
        self._gcon.commit()

    # ── loaders ────────────────────────────────────────────────────────────────
    def _chunk_timestamps(self) -> dict[int, datetime | None]:
        out: dict[int, datetime | None] = {}
        if not self.meta_path.exists():
            return out
        con = sqlite3.connect(f"file:{self.meta_path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        for r in con.execute("SELECT chunk_id, meta_json FROM chunks"):
            meta = json.loads(r["meta_json"])
            out[r["chunk_id"]] = parse_timestamp(meta.get("ingested_at"))
        con.close()
        return out

    def _entities(self) -> dict[int, dict]:
        return {
            r["entity_id"]: json.loads(r["node_json"])
            for r in self._gcon.execute("SELECT entity_id, node_json FROM entities")
        }

    def _relations(self) -> list[dict]:
        return [json.loads(r["rel_json"]) for r in self._gcon.execute("SELECT rel_json FROM relations")]

    def _entity_chunks(self) -> dict[int, list[int]]:
        return {
            r["entity_id"]: json.loads(r["chunk_ids"])
            for r in self._gcon.execute("SELECT entity_id, chunk_ids FROM entity_chunks")
        }

    def _chunk_entities(self) -> dict[int, list[int]]:
        return {
            r["chunk_id"]: json.loads(r["entity_ids"])
            for r in self._gcon.execute("SELECT chunk_id, entity_ids FROM chunk_entities")
        }

    @staticmethod
    def _max_ts(chunk_ids: list[int], ts: dict[int, datetime | None]) -> datetime | None:
        stamps = [ts[c] for c in chunk_ids if ts.get(c) is not None]
        return max(stamps) if stamps else None

    # ── core computation ─────────────────────────────────────────────────────
    def consolidate(self) -> dict:
        ts = self._chunk_timestamps()
        entities = self._entities()
        entity_chunks = self._entity_chunks()
        chunk_entities = self._chunk_entities()
        relations = self._relations()

        node_rows = []
        node_state = Counter()
        node_strength: dict[int, float] = {}
        for eid, node in entities.items():
            reinforcement = int(node.get("mention_count", 1))
            last_seen = self._max_ts(entity_chunks.get(eid, []), ts)
            el = elapsed_days(last_seen, self.now)
            s = strength(reinforcement, el, self.params)
            state = classify(s, self.params)
            node_strength[eid] = s
            node_state[state] += 1
            node_rows.append((
                eid, node.get("name", ""), node.get("entity_type", "Unknown"),
                reinforcement, last_seen.isoformat() if last_seen else None,
                stability(reinforcement, self.params), s, state,
            ))

        edge_rows = []
        edge_state = Counter()
        for rel in relations:
            evid = rel.get("evidence_chunk_ids", [])
            reinforcement = max(1, len(evid))
            last_seen = self._max_ts(evid, ts)
            el = elapsed_days(last_seen, self.now)
            s = strength(reinforcement, el, self.params)
            state = classify(s, self.params)
            edge_state[state] += 1
            edge_rows.append((
                rel["src_id"], rel["dst_id"], rel["relation_type"], reinforcement,
                last_seen.isoformat() if last_seen else None, s, state,
            ))

        chunk_rows = []
        chunk_state = Counter()
        for cid, ent_ids in chunk_entities.items():
            reinforcement = max(1, len(ent_ids))
            last_seen = ts.get(cid)
            el = elapsed_days(last_seen, self.now)
            s = strength(reinforcement, el, self.params)
            state = classify(s, self.params)
            chunk_state[state] += 1
            chunk_rows.append((
                cid, reinforcement, last_seen.isoformat() if last_seen else None, s, state,
            ))

        with self._gcon:
            self._gcon.execute("DELETE FROM memory_nodes")
            self._gcon.execute("DELETE FROM memory_edges")
            self._gcon.execute("DELETE FROM memory_chunks")
            self._gcon.executemany(
                "INSERT INTO memory_nodes VALUES (?,?,?,?,?,?,?,?)", node_rows)
            self._gcon.executemany(
                "INSERT INTO memory_edges VALUES (?,?,?,?,?,?,?)", edge_rows)
            self._gcon.executemany(
                "INSERT INTO memory_chunks VALUES (?,?,?,?,?)", chunk_rows)

        return {
            "nodes": dict(node_state),
            "edges": dict(edge_state),
            "chunks": dict(chunk_state),
            "node_strength": node_strength,
        }

    def synthesize_core_facts(
        self, node_strength: dict[int, float], top_k: int = 10,
        model: str = "llama3.1:8b", base_url: str = "http://localhost:11434",
    ) -> int:
        entities = self._entities()
        # Build outgoing adjacency for description.
        adj: dict[int, list[str]] = {}
        for rel in self._relations():
            src = entities.get(rel["src_id"])
            dst = entities.get(rel["dst_id"])
            if src and dst:
                adj.setdefault(rel["src_id"], []).append(f"{rel['relation_type']} {dst.get('name','')}")

        top = sorted(node_strength, key=lambda e: node_strength[e], reverse=True)[:top_k]
        written = 0
        with self._gcon:
            self._gcon.execute("DELETE FROM consolidated_facts")
            for eid in top:
                node = entities.get(eid)
                if not node:
                    continue
                rels = adj.get(eid, [])[:8]
                if not rels:
                    continue
                prompt = (
                    f"Entity: {node.get('name','')} ({node.get('entity_type','')})\n"
                    f"Known relations: {'; '.join(rels)}\n\n"
                    "Write ONE concise factual sentence consolidating what this entity is and "
                    "its most important connections."
                )
                try:
                    fact = _ollama_chat(prompt, model, base_url)
                except urllib.error.URLError as e:
                    print(f"  (synthesis skipped, Ollama unreachable: {e})", file=sys.stderr)
                    break
                if fact:
                    self._gcon.execute(
                        "INSERT OR REPLACE INTO consolidated_facts VALUES (?,?,?)",
                        (eid, fact, node_strength[eid]),
                    )
                    written += 1
        return written

    # ── readers ────────────────────────────────────────────────────────────────
    def top_nodes(self, limit: int = 15) -> list[sqlite3.Row]:
        return list(self._gcon.execute(
            "SELECT name, entity_type, reinforcement, strength, state "
            "FROM memory_nodes ORDER BY strength DESC LIMIT ?", (limit,)))

    def chunk_state_counts(self) -> dict:
        return {
            r["state"]: r["n"]
            for r in self._gcon.execute(
                "SELECT state, COUNT(*) n FROM memory_chunks GROUP BY state")
        }

    def facts(self, limit: int = 20) -> list[sqlite3.Row]:
        return list(self._gcon.execute(
            "SELECT fact, strength FROM consolidated_facts ORDER BY strength DESC LIMIT ?", (limit,)))

    def close(self) -> None:
        self._gcon.close()


# ── CLI ────────────────────────────────────────────────────────────────────────
def _parse_now(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise SystemExit(f"--now must be ISO format (e.g. 2026-12-31), got: {value}")
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _cmd_run(args: argparse.Namespace) -> int:
    params = MemoryParams(
        base_stability_days=args.stability,
        promote_threshold=args.promote,
        demote_threshold=args.demote,
    )
    loop = DreamLoop(store_dir=Path(args.store_dir), params=params, now=_parse_now(args.now))
    print(f"Dreaming over {loop.graph_path.name} (now={loop.now.date()}, "
          f"stability={params.base_stability_days}d) ...")
    stats = loop.consolidate()
    print(f"  nodes : {stats['nodes']}")
    print(f"  edges : {stats['edges']}")
    print(f"  chunks: {stats['chunks']}")
    n_dormant = stats["chunks"].get(DORMANT, 0)
    print(f"  -> demoted {n_dormant} chunk(s) to dormant (kept, deprioritized)")

    if args.synthesize:
        print("  synthesizing core facts from strongest entities ...")
        n = loop.synthesize_core_facts(
            stats["node_strength"], top_k=args.top_k,
            model=args.llm_model, base_url=args.ollama_url)
        print(f"  wrote {n} consolidated core fact(s)")
    loop.close()
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    loop = DreamLoop(store_dir=Path(args.store_dir))
    if args.what == "nodes":
        print(f"{'strength':>8}  {'state':<10} {'reinf':>5}  name")
        for r in loop.top_nodes(args.limit):
            print(f"{r['strength']:8.3f}  {r['state']:<10} {r['reinforcement']:5d}  "
                  f"{r['name']} [{r['entity_type']}]")
    elif args.what == "chunks":
        print("chunk states:", loop.chunk_state_counts())
    elif args.what == "facts":
        rows = loop.facts(args.limit)
        if not rows:
            print("No consolidated facts yet — run `dream.py run --synthesize`.")
        for r in rows:
            print(f"[{r['strength']:.3f}] {r['fact']}")
    loop.close()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dream", description="Dream RAG memory consolidation loop")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Recompute memory strengths and consolidate")
    run.add_argument("--store-dir", default=str(DEFAULT_STORE_DIR))
    run.add_argument("--now", default=None,
                     help="simulate the current date (ISO, e.g. 2027-01-01) for the forgetting curve")
    run.add_argument("--stability", type=float, default=30.0, help="base stability in days")
    run.add_argument("--promote", type=float, default=0.66)
    run.add_argument("--demote", type=float, default=0.33)
    run.add_argument("--synthesize", action="store_true", help="LLM-synthesize core facts")
    run.add_argument("--top-k", type=int, default=10)
    run.add_argument("--llm-model", default="llama3.1:8b")
    run.add_argument("--ollama-url", default="http://localhost:11434")

    show = sub.add_parser("show", help="Inspect memory state")
    show.add_argument("what", choices=["nodes", "chunks", "facts"])
    show.add_argument("--store-dir", default=str(DEFAULT_STORE_DIR))
    show.add_argument("--limit", type=int, default=15)

    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        raise SystemExit(_cmd_run(args))
    if args.command == "show":
        raise SystemExit(_cmd_show(args))


if __name__ == "__main__":
    main()
