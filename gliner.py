"""
gliner.py — Thin async HTTP client for a GLiNER NER server.

GLiNER is a compact encoder model (~0.5B) that runs on CPU in ~5ms/chunk.
This client treats it as an advisory pre-screener: detected Person spans are
injected into the LLM prompt as high-confidence hints. It also supports
multi-label extraction for graph / hybrid retrieval.

Expected API (scripts/gliner_server.py):
  POST /ner
  Request:  {"text": "...", "labels": ["person"], "threshold": 0.4}
  Response: [{"text": "Joe Rassool", "label": "person", "score": 0.92}, ...]

All errors are soft failures — extract methods return [] rather than raising.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import aiohttp

from gliner_ner import DEFAULT_LABELS, GlinerSpan, map_label

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

    async def extract_entities(
        self,
        text: str,
        labels: Optional[Sequence[str]] = None,
    ) -> list[GlinerSpan]:
        """Return typed entity spans above the configured threshold.

        Never raises — any failure degrades to an empty list.
        """
        body = {
            "text": text,
            "labels": list(labels) if labels is not None else list(DEFAULT_LABELS),
            "threshold": self.threshold,
        }
        try:
            session = self._get_session()
            async with session.post(f"{self.url}/ner", json=body) as resp:
                if not resp.ok:
                    logger.warning(
                        "GLiNER request failed with HTTP %s (continuing without hints)",
                        resp.status,
                    )
                    return []
                data = await resp.json()
        except Exception as e:
            logger.warning("GLiNER request failed (continuing without hints): %s", e)
            return []

        if not isinstance(data, list):
            logger.warning("GLiNER response is not a JSON array")
            return []

        spans: list[GlinerSpan] = []
        seen: set[tuple[str, str]] = set()
        for item in data:
            if not isinstance(item, dict):
                continue
            span_text = item.get("text")
            label = item.get("label")
            score = item.get("score")
            if not isinstance(span_text, str) or not span_text.strip():
                continue
            if not isinstance(label, str):
                continue
            if not isinstance(score, (int, float)) or score < self.threshold:
                continue
            key = (span_text.strip().lower(), label.lower())
            if key in seen:
                continue
            seen.add(key)
            spans.append(
                GlinerSpan(text=span_text.strip(), label=label, score=float(score))
            )
        spans.sort(key=lambda s: s.score, reverse=True)
        return spans

    async def person_spans(self, text: str) -> list[str]:
        """Return unique Person span texts detected above the configured threshold.

        Never raises — any failure degrades to an empty hint list.
        """
        spans = await self.extract_entities(text, labels=["person"])
        return sorted({s.text for s in spans if map_label(s.label) == "Person"})
