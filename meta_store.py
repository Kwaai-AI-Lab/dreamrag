"""
meta_store.py — Per-tenant chunk metadata and sync metadata store.

Mirrors meta_store.rs. Uses SQLite (via stdlib sqlite3) instead of redb.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID


@dataclass
class SyncMeta:
    file_path: str
    mtime_secs: int
    file_size: int


@dataclass
class ChunkMeta:
    doc_name: str
    chunk_index: int
    text: str
    surrounding: str
    page_num: Optional[int]
    ingested_at: str
    section_name: Optional[str] = None
    skip_extraction: bool = False
    section_note: Optional[str] = None
    section_type: str = "main"


class MetaStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._con = sqlite3.connect(self._db_path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._init_tables()

    @classmethod
    def open(cls, data_dir: str | Path, tenant_id: UUID) -> "MetaStore":
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = data_dir / f"{tenant_id}.db"
        return cls(db_path)

    def _init_tables(self) -> None:
        self._con.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id  INTEGER PRIMARY KEY,
                doc_name  TEXT NOT NULL,
                meta_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_name);

            CREATE TABLE IF NOT EXISTS docs (
                doc_name  TEXT PRIMARY KEY,
                chunk_ids TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sync (
                doc_name  TEXT PRIMARY KEY,
                meta_json TEXT NOT NULL
            );
        """)
        self._con.commit()

    def put_chunks(self, chunks: list[ChunkMeta | dict], ids: list[int]) -> None:
        assert len(chunks) == len(ids)
        doc_ids: dict[str, list[int]] = {}
        rows = []
        for meta, cid in zip(chunks, ids):
            if isinstance(meta, dict):
                doc_name = meta["doc_name"]
                meta_json = json.dumps(meta)
            else:
                doc_name = meta.doc_name
                meta_json = json.dumps(asdict(meta))
            rows.append((cid, doc_name, meta_json))
            doc_ids.setdefault(doc_name, []).append(cid)

        with self._con:
            self._con.executemany(
                "INSERT OR REPLACE INTO chunks(chunk_id, doc_name, meta_json) VALUES (?,?,?)",
                rows,
            )
            for doc_name, new_ids in doc_ids.items():
                row = self._con.execute(
                    "SELECT chunk_ids FROM docs WHERE doc_name=?", (doc_name,)
                ).fetchone()
                existing = json.loads(row["chunk_ids"]) if row else []
                merged = sorted(set(existing) | set(new_ids))
                self._con.execute(
                    "INSERT OR REPLACE INTO docs(doc_name, chunk_ids) VALUES (?,?)",
                    (doc_name, json.dumps(merged)),
                )

    def get_chunks(self, ids: list[int]) -> list[Optional[ChunkMeta]]:
        out = []
        for cid in ids:
            row = self._con.execute(
                "SELECT meta_json FROM chunks WHERE chunk_id=?", (cid,)
            ).fetchone()
            if row:
                data = json.loads(row["meta_json"])
                out.append(ChunkMeta(**data))
            else:
                out.append(None)
        return out

    def all_chunks(self) -> list[tuple[int, ChunkMeta]]:
        rows = self._con.execute("SELECT chunk_id, meta_json FROM chunks").fetchall()
        return [(r["chunk_id"], ChunkMeta(**json.loads(r["meta_json"]))) for r in rows]

    def list_docs(self) -> list[str]:
        rows = self._con.execute("SELECT doc_name FROM docs ORDER BY doc_name").fetchall()
        return [r["doc_name"] for r in rows]

    def delete_doc(self, doc_name: str) -> list[int]:
        row = self._con.execute(
            "SELECT chunk_ids FROM docs WHERE doc_name=?", (doc_name,)
        ).fetchone()
        if not row:
            return []
        ids: list[int] = json.loads(row["chunk_ids"])
        with self._con:
            for cid in ids:
                self._con.execute("DELETE FROM chunks WHERE chunk_id=?", (cid,))
            self._con.execute("DELETE FROM docs WHERE doc_name=?", (doc_name,))
        return ids

    @staticmethod
    def now_rfc3339() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Sync metadata ─────────────────────────────────────────────────────────

    def put_sync_meta(self, doc_name: str, meta: SyncMeta) -> None:
        with self._con:
            self._con.execute(
                "INSERT OR REPLACE INTO sync(doc_name, meta_json) VALUES (?,?)",
                (doc_name, json.dumps(asdict(meta))),
            )

    def get_sync_meta(self, doc_name: str) -> Optional[SyncMeta]:
        row = self._con.execute(
            "SELECT meta_json FROM sync WHERE doc_name=?", (doc_name,)
        ).fetchone()
        if not row:
            return None
        return SyncMeta(**json.loads(row["meta_json"]))

    def delete_sync_meta(self, doc_name: str) -> None:
        with self._con:
            self._con.execute("DELETE FROM sync WHERE doc_name=?", (doc_name,))

    def all_sync_metas(self) -> list[tuple[str, SyncMeta]]:
        rows = self._con.execute("SELECT doc_name, meta_json FROM sync").fetchall()
        return [(r["doc_name"], SyncMeta(**json.loads(r["meta_json"]))) for r in rows]

    def close(self) -> None:
        self._con.close()
