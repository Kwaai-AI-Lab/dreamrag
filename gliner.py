"""
gliner.py — Thin async HTTP client for a GLiNER NER server.

GLiNER is a compact encoder model (~0.5B) that runs on CPU in ~5ms/chunk.
This client treats it as an advisory pre-screener: detected Person spans are
injected into the LLM prompt as high-confidence hints.

Expected API (scripts/gliner_server.py):
  POST /ner
  Request:  {"text": "...", "labels": ["person"], "threshold": 0.4}
  Response: [{"text": "Joe Rassool", "label": "person", "score": 0.92}, ...]

All errors are soft failures — person_spans() returns [] rather than raising.
"""

from __future__ import annotations

import logging
import aiohttp

logger = logging.getLogger(__name__)


class GliNERClient:
    """Async HTTP client for a running GLiNER server."""

    def __init__(self, url: str, threshold: float = 0.4) -> None:
        self.url = url.rstrip("/")
        self.threshold = threshold
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def person_spans(self, text: str) -> list[str]:
        """Return unique Person span texts detected above the configured threshold.

        Never raises — any failure degrades to an empty hint list.
        """
        body = {
            "text": text,
            "labels": ["person"],
            "threshold": self.threshold,
        }
        try:
            session = self._get_session()
            async with session.post(f"{self.url}/ner", json=body) as resp:
                if not resp.ok:
                    logger.warning("GLiNER request failed with HTTP %s (continuing without hints)", resp.status)
                    return []
                data = await resp.json()
        except Exception as e:
            logger.warning("GLiNER request failed (continuing without hints): %s", e)
            return []

        if not isinstance(data, list):
            logger.warning("GLiNER response is not a JSON array")
            return []

        spans = [
            item["text"]
            for item in data
            if isinstance(item, dict)
            and isinstance(item.get("score"), (int, float))
            and item["score"] >= self.threshold
            and isinstance(item.get("text"), str)
        ]
        return sorted(set(spans))
