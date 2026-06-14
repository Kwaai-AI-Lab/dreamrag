"""
embedder.py — Async HTTP client for Ollama embedding API.

Mirrors embedder.rs. Default model: nomic-embed-text (768-dim).
Adds asymmetric instruction prefixes for nomic models to improve retrieval.
"""

from __future__ import annotations

import os
import logging
import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_DIM = 768

NOMIC_QUERY_PREFIX = "search_query: "
NOMIC_DOC_PREFIX = "search_document: "


class EmbedClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def _is_nomic(self) -> bool:
        return "nomic" in self.model

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single search query (adds search_query: prefix for nomic)."""
        if self._is_nomic():
            text = f"{NOMIC_QUERY_PREFIX}{text}"
        results = await self._embed_raw([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks (adds search_document: prefix for nomic)."""
        if self._is_nomic():
            texts = [f"{NOMIC_DOC_PREFIX}{t}" for t in texts]
        return await self._embed_raw(texts)

    async def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/api/embed"
        body = {"model": self.model, "input": texts}
        session = self._get_session()
        try:
            async with session.post(url, json=body) as resp:
                if not resp.ok:
                    text_body = await resp.text()
                    raise RuntimeError(f"Ollama embed error {resp.status}: {text_body}")
                data = await resp.json()
        except aiohttp.ClientError as e:
            raise RuntimeError(f"POST {url} — is Ollama running? {e}") from e
        return data["embeddings"]

    async def probe_dim(self) -> int:
        """Probe Ollama and return the embedding dimension."""
        results = await self._embed_raw(["probe"])
        return len(results[0])

    async def check_dim_matches(self, expected: int) -> None:
        """Verify the embedding dimension matches expected. Call at startup."""
        actual = await self.probe_dim()
        if actual != expected:
            raise RuntimeError(
                f"Embedding model '{self.model}' returns {actual} dimensions "
                f"but KB was initialised with {expected}. "
                f"Destroy and re-init the KB with the correct model."
            )
