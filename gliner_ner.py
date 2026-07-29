"""
gliner_ner.py — High-quality multi-label NER via GLiNER.

Primary path: HTTP client against scripts/gliner_server.py.
Optional fallback: in-process GLiNER if the package is installed.
Soft-fails to [] when neither is available so callers can fall back to spaCy/patterns.

Includes post-filters to drop generic nouns, OCR junk, and low-confidence type noise.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_LABELS: list[str] = [
    "person",
    "organization",
    "location",
    "work",
    "event",
    "facility",
]

LABEL_TO_TYPE = {
    "person": "Person",
    "organization": "Organization",
    "org": "Organization",
    "location": "Location",
    "gpe": "Location",
    "loc": "Location",
    "product": "Product",
    "event": "Event",
    "work": "Work",
    "work of art": "Work",
    "facility": "Facility",
    "fac": "Facility",
    "date": "Date",
}

# Per-type score floors (on top of the global threshold).
TYPE_MIN_SCORE = {
    "Person": 0.55,
    "Organization": 0.55,
    "Location": 0.60,
    "Work": 0.65,
    "Event": 0.65,
    "Facility": 0.60,
    "Product": 0.75,
}

DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_THRESHOLD = 0.55

# Generic / meta words that GLiNER sometimes labels as entities.
GENERIC_BLOCKLIST = {
    "cluster", "paper", "review", "article", "document", "chapter", "section",
    "literature", "model", "framework", "system", "method", "approach",
    "house", "home", "court", "defendant", "plaintiff", "company", "organization",
    "university", "institute", "school", "department", "agency", "government",
    "person", "people", "man", "woman", "child", "children", "author", "writer",
    "grass", "leaves", "book", "poem", "poems", "novel", "story", "text",
    "node", "way", "relation", "tag", "tags", "map", "data",
    "purpose", "role", "version", "reference", "tutorial", "module",
    "function", "functions", "object", "type", "types", "class", "classes",
    "import", "scope", "namespace", "grammar", "statement", "statements",
    "near-miss", "cross-document", "observational", "uncertainty", "ensemble",
    "scholarly article", "discovery paper", "review paper",
    "whaling-industry literature", "historical event", "1820 historical event",
}

COMMON_SINGLE_TOKENS = {
    "the", "and", "or", "but", "that", "this", "from", "with", "for", "are",
    "was", "were", "been", "being", "have", "has", "which", "who", "what",
    "when", "where", "why", "how", "into", "over", "under", "about", "also",
    "such", "than", "then", "them", "they", "their", "there", "these", "those",
    "one", "two", "three", "first", "second", "new", "old", "good", "great",
    "small", "large", "high", "low", "same", "other", "many", "most", "some",
    "any", "all", "each", "every", "own", "more", "less", "very", "just",
    "only", "even", "still", "already", "often", "never", "always", "yes", "no",
    "true", "false", "null", "none", "unknown", "n/a", "etc",
}

_inprocess_model = None
_inprocess_model_name: Optional[str] = None


@dataclass
class GlinerSpan:
    """A single GLiNER entity span."""

    text: str
    label: str
    score: float

    @property
    def entity_type(self) -> str:
        return LABEL_TO_TYPE.get(self.label.lower().strip(), self.label.title())


def map_label(label: str) -> str:
    """Map a GLiNER label to DreamRAG entity types."""
    return LABEL_TO_TYPE.get(label.lower().strip(), label.title())


def normalize_entity_name(text: str) -> str:
    """Collapse OCR/whitespace junk into a clean entity surface form."""
    cleaned = text.replace("\u00a0", " ")
    cleaned = cleaned.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip("\"'`“”‘’[](){}<>|")
    cleaned = re.sub(r"[;,:]+$", "", cleaned).strip()
    return cleaned


def is_valid_entity(name: str, entity_type: str, score: float = 1.0) -> bool:
    """Return False for generic, noisy, or low-confidence spans."""
    if not name:
        return False

    min_score = TYPE_MIN_SCORE.get(entity_type, DEFAULT_THRESHOLD)
    if score < min_score:
        return False

    if len(name) < 2 or len(name) > 80:
        return False

    alpha = sum(ch.isalpha() for ch in name)
    if alpha < 2 or alpha / max(1, len(name)) < 0.45:
        return False

    lower = name.lower().strip()
    if lower in GENERIC_BLOCKLIST or lower in COMMON_SINGLE_TOKENS:
        return False

    for blocked in GENERIC_BLOCKLIST:
        if " " in blocked and blocked in lower and len(lower) <= len(blocked) + 8:
            return False

    tokens = name.split()
    if entity_type == "Person":
        if len(tokens) == 1 and not tokens[0][:1].isupper():
            return False
        if lower in {"grass", "typee", "leaves", "house", "defendant", "plaintiff"}:
            return False
        if len(tokens) > 5:
            return False

    if entity_type == "Location":
        if lower in {"cluster", "paper", "section", "chapter", "home", "house"}:
            return False
        if len(tokens) == 1 and lower.endswith(("ing", "tion", "ment", "ness")):
            return False

    if entity_type in {"Work", "Event", "Product"}:
        if len(tokens) == 1 and len(name) < 4 and not name.isupper():
            return False

    if "@" in name:
        return False
    if entity_type != "Organization" and re.fullmatch(
        r"[a-z0-9.-]+\.(com|org|net|edu|io|gov)", lower
    ):
        return False

    return True


def clean_spans(
    spans: Sequence[GlinerSpan], threshold: float = DEFAULT_THRESHOLD
) -> List[GlinerSpan]:
    """Normalize, filter, and de-overlap GLiNER spans."""
    cleaned: List[GlinerSpan] = []
    for span in spans:
        name = normalize_entity_name(span.text)
        if not name:
            continue
        entity_type = span.entity_type
        score = float(span.score)
        if score < threshold:
            continue
        if not is_valid_entity(name, entity_type, score):
            continue
        cleaned.append(GlinerSpan(text=name, label=span.label, score=score))

    cleaned.sort(key=lambda s: (s.score, len(s.text)), reverse=True)
    kept: List[GlinerSpan] = []
    seen_keys: set[tuple[str, str]] = set()
    for span in cleaned:
        key = (span.text.lower(), span.entity_type)
        if key in seen_keys:
            continue
        dominated = False
        for other in kept:
            if other.entity_type != span.entity_type:
                continue
            a, b = other.text.lower(), span.text.lower()
            if a == b:
                dominated = True
                break
            if b in a and len(a) - len(b) <= 20 and span.score <= other.score + 0.05:
                dominated = True
                break
        if dominated:
            continue
        seen_keys.add(key)
        kept.append(span)

    kept.sort(key=lambda s: s.score, reverse=True)
    return kept


def extract_entities(
    text: str,
    *,
    url: Optional[str] = DEFAULT_URL,
    labels: Optional[Sequence[str]] = None,
    threshold: float = DEFAULT_THRESHOLD,
    prefer_inprocess: bool = False,
) -> List[GlinerSpan]:
    """Extract entities with GLiNER.

    Tries HTTP server first (unless prefer_inprocess=True), then in-process model.
    Returns [] on soft failure.
    """
    if not text or not text.strip():
        return []

    label_list = list(labels) if labels is not None else list(DEFAULT_LABELS)

    if prefer_inprocess:
        spans = _extract_inprocess(text, label_list, threshold)
        if spans:
            return clean_spans(spans, threshold=threshold)
        if url:
            http_spans = _extract_http(text, url, label_list, threshold)
            return clean_spans(http_spans or [], threshold=threshold)
        return []

    if url:
        spans = _extract_http(text, url, label_list, threshold)
        if spans is not None:
            return clean_spans(spans, threshold=threshold)

    return clean_spans(
        _extract_inprocess(text, label_list, threshold), threshold=threshold
    )


def health_check(url: str = DEFAULT_URL, timeout: float = 2.0) -> bool:
    """Return True if the GLiNER HTTP server is reachable."""
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/health", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def inprocess_available() -> bool:
    """Return True if the GLiNER pip package can be imported (not local gliner.py)."""
    return _import_gliner_package() is not None


def _import_gliner_package():
    """Import the pip `gliner` package, avoiding shadowing by local gliner.py."""
    import importlib
    import sys
    from pathlib import Path

    project_root = str(Path(__file__).resolve().parent)
    removed: list[tuple[int, str]] = []
    for i, entry in list(enumerate(sys.path)):
        if entry in ("", ".") or entry == project_root or Path(entry).resolve() == Path(project_root):
            removed.append((i, entry))
    for _, entry in reversed(removed):
        try:
            sys.path.remove(entry)
        except ValueError:
            pass

    cached = sys.modules.pop("gliner", None)
    try:
        mod = importlib.import_module("gliner")
        if not hasattr(mod, "GLiNER"):
            return None
        return mod
    except ImportError:
        return None
    finally:
        for i, entry in removed:
            if entry not in sys.path:
                sys.path.insert(min(i, len(sys.path)), entry)
        if "gliner" not in sys.modules and cached is not None and not hasattr(cached, "GLiNER"):
            sys.modules["gliner"] = cached


def _extract_http(
    text: str,
    url: str,
    labels: Sequence[str],
    threshold: float,
) -> Optional[List[GlinerSpan]]:
    """POST to /ner. Returns [] for empty results, None if the server failed."""
    server_threshold = max(0.35, min(threshold, 0.45))
    body = json.dumps(
        {
            "text": text,
            "labels": list(labels),
            "threshold": server_threshold,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{url.rstrip('/')}/ner",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        logger.debug("GLiNER HTTP NER failed: %s", e)
        return None
    except Exception as e:
        logger.warning("GLiNER HTTP NER unexpected error: %s", e)
        return None

    if not isinstance(data, list):
        logger.warning("GLiNER response is not a JSON array")
        return None

    return _parse_spans(data, server_threshold)


def _extract_inprocess(
    text: str,
    labels: Sequence[str],
    threshold: float,
    model_name: str = "urchade/gliner_small-v2.1",
) -> List[GlinerSpan]:
    """Run GLiNER in-process if the package is installed."""
    global _inprocess_model, _inprocess_model_name

    gliner_mod = _import_gliner_package()
    if gliner_mod is None:
        logger.debug("gliner package not installed; skipping in-process NER")
        return []

    try:
        if _inprocess_model is None or _inprocess_model_name != model_name:
            logger.info("Loading in-process GLiNER model: %s", model_name)
            _inprocess_model = gliner_mod.GLiNER.from_pretrained(model_name)
            _inprocess_model_name = model_name

        server_threshold = max(0.35, min(threshold, 0.45))
        entities = _inprocess_model.predict_entities(
            text, list(labels), threshold=server_threshold
        )
        return _parse_spans(entities, server_threshold)
    except Exception as e:
        logger.warning("In-process GLiNER failed: %s", e)
        return []


def _parse_spans(data: list, threshold: float) -> List[GlinerSpan]:
    spans: List[GlinerSpan] = []
    seen: set[tuple[str, str]] = set()

    for item in data:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        label = item.get("label")
        score = item.get("score")
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(label, str):
            continue
        if not isinstance(score, (int, float)):
            continue
        if score < threshold:
            continue

        name = normalize_entity_name(text)
        if not name:
            continue
        key = (name.lower(), label.lower())
        if key in seen:
            continue
        seen.add(key)
        spans.append(GlinerSpan(text=name, label=label, score=float(score)))

    spans.sort(key=lambda s: s.score, reverse=True)
    return spans


if __name__ == "__main__":
    import sys

    sample = " ".join(sys.argv[1:]) or (
        "Marie Curie worked at the University of Paris and discovered radium in France."
    )
    print(f"text: {sample}")
    print(f"server healthy: {health_check()}")
    for span in extract_entities(sample):
        print(f"  {span.text!r:30} {span.entity_type:15} score={span.score:.3f}")
