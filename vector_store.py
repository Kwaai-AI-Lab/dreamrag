"""
vector_store.py — SQLite-backed store for chunk embedding vectors.
"""

from __future__ import annotations

import math
import sqlite3
import struct
from pathlib import Path
from uuid import UUID


class VectorStore:
    def __init__(self, db_path: str | Path, dim: int) -> None:
        self._db_path = str(db_path)
        self.dim = dim
        self._con = sqlite3.connect(self._db_path, check_same_thread=False)
        self._init_tables()

    @classmethod
    def open(cls, data_dir: str | Path, tenant_id: UUID, dim: int) -> "VectorStore":
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        return cls(data_dir / f"vectors-{tenant_id}.db", dim)

    @classmethod
    def open_existing(cls, data_dir: str | Path, tenant_id: UUID, default_dim: int = 768) -> "VectorStore":
        data_dir = Path(data_dir)
        db_path = data_dir / f"vectors-{tenant_id}.db"
        if not db_path.exists():
            return cls(db_path, default_dim)
        con = sqlite3.connect(db_path)
        row = con.execute("SELECT value FROM meta WHERE key = 'dim'").fetchone()
        con.close()
        dim = int(row[0]) if row else default_dim
        return cls(db_path, dim)

    def _init_tables(self) -> None:
        self._con.executescript("""
            CREATE TABLE IF NOT EXISTS vectors (
                chunk_id   INTEGER PRIMARY KEY,
                embedding  BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        self._con.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("dim", str(self.dim)),
        )
        self._con.commit()

    def _pack(self, embedding: list[float]) -> bytes:
        if len(embedding) != self.dim:
            raise ValueError(f"expected {self.dim}-dim vector, got {len(embedding)}")
        return struct.pack(f"<{self.dim}f", *embedding)

    def _unpack(self, blob: bytes) -> list[float]:
        return list(struct.unpack(f"<{self.dim}f", blob))

    def upsert_batch(self, vectors: list[tuple[int, list[float]]]) -> int:
        rows = [(cid, self._pack(emb)) for cid, emb in vectors]
        with self._con:
            self._con.executemany(
                "INSERT OR REPLACE INTO vectors(chunk_id, embedding) VALUES (?, ?)",
                rows,
            )
        return len(rows)

    def delete(self, chunk_ids: list[int]) -> int:
        if not chunk_ids:
            return 0
        with self._con:
            self._con.executemany(
                "DELETE FROM vectors WHERE chunk_id = ?",
                [(cid,) for cid in chunk_ids],
            )
        return len(chunk_ids)

    def count(self) -> int:
        row = self._con.execute("SELECT COUNT(*) FROM vectors").fetchone()
        return int(row[0]) if row else 0

    def search(self, query_emb: list[float], top_k: int = 5) -> list[tuple[int, float]]:
        """Return (chunk_id, cosine_similarity) pairs, highest first."""
        if len(query_emb) != self.dim:
            raise ValueError(f"expected {self.dim}-dim query, got {len(query_emb)}")

        qnorm = math.sqrt(sum(x * x for x in query_emb))
        if qnorm == 0:
            return []

        scores: list[tuple[int, float]] = []
        for row in self._con.execute("SELECT chunk_id, embedding FROM vectors"):
            emb = self._unpack(row[1])
            dot = sum(a * b for a, b in zip(query_emb, emb))
            norm = math.sqrt(sum(x * x for x in emb))
            if norm == 0:
                continue
            scores.append((row[0], dot / (qnorm * norm)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def close(self) -> None:
        self._con.close()
