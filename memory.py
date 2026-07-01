"""
memory.py — Memory-strength model for the Dream RAG knowledge graph.

Implements the "memory paradigm" pillar: every node, edge, and chunk has a
*strength* that combines reinforcement (how often it was seen) and recency (how
long ago), decaying according to a dynamically-modulated Ebbinghaus forgetting
curve. Reinforcement increases the memory's stability, so well-supported facts
decay slowly (the spacing effect) while one-off mentions fade quickly.

    stability S = base_stability_days * (1 + boost * ln(1 + reinforcement))
    retention  R = exp(-elapsed_days / S)            # Ebbinghaus
    strength     = R                                  # in (0, 1]

State is then thresholded:
    strength >= promote_threshold  -> long_term  (consolidated / core)
    strength >= demote_threshold   -> short_term (kept, deprioritized)
    else                           -> dormant    (forgettable)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

LONG_TERM = "long_term"
SHORT_TERM = "short_term"
DORMANT = "dormant"


@dataclass
class MemoryParams:
    base_stability_days: float = 30.0
    boost: float = 1.5
    promote_threshold: float = 0.66
    demote_threshold: float = 0.33


def parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def elapsed_days(last_seen: datetime | None, now: datetime | None = None) -> float:
    if last_seen is None:
        return 0.0
    now = now or datetime.now(timezone.utc)
    if last_seen.tzinfo is None:
        last_seen = last_seen.replace(tzinfo=timezone.utc)
    delta = now - last_seen
    return max(0.0, delta.total_seconds() / 86400.0)


def stability(reinforcement: int, params: MemoryParams) -> float:
    return params.base_stability_days * (1.0 + params.boost * math.log1p(max(0, reinforcement)))


def retention(elapsed: float, stability_days: float) -> float:
    if stability_days <= 0:
        return 0.0
    return math.exp(-elapsed / stability_days)


def strength(reinforcement: int, elapsed: float, params: MemoryParams) -> float:
    return retention(elapsed, stability(reinforcement, params))


def classify(strength_value: float, params: MemoryParams) -> str:
    if strength_value >= params.promote_threshold:
        return LONG_TERM
    if strength_value >= params.demote_threshold:
        return SHORT_TERM
    return DORMANT
