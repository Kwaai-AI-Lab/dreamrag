"""
ner.py — Lightweight proper-noun pre-screening and pronoun resolution.

Mirrors ner.rs. Pure Python, no external NLP dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

CANDIDATE_CAP = 40

STOP_WORDS = {
    "The", "A", "An", "This", "That", "These", "Those", "Its", "It",
    "He", "Him", "His", "She", "Her", "Hers", "They", "Them", "Their",
    "Theirs", "We", "Our", "You", "Your", "I", "My", "Me",
    "In", "On", "At", "By", "For", "From", "With", "Of", "About", "And",
    "Or", "But", "If", "As", "So", "Yet", "Nor", "Both", "Also", "To",
    "Into", "Up", "Between", "Among", "Before", "After", "During",
    "Through", "Within", "Over",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
}

MALE_PRONOUNS = {"he", "him", "his", "himself"}
FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}
NEUTRAL_PRONOUNS = {"they", "them", "their", "theirs", "themselves"}


def _core(word: str) -> str:
    return word.strip("".join(c for c in word if not c.isalnum()))


def _strip_punctuation(word: str) -> str:
    result = word
    while result and not result[0].isalnum():
        result = result[1:]
    while result and not result[-1].isalnum():
        result = result[:-1]
    return result


def _ends_phrase(word: str) -> bool:
    return bool(word) and word[-1] in ".!?,;:)\"'"


def _is_stop_word(s: str) -> bool:
    return s in STOP_WORDS


def extract_proper_noun_candidates(text: str) -> list[str]:
    """Extract proper noun candidate phrases from text."""
    seen: set[str] = set()
    result: list[str] = []
    words = text.split()
    n = len(words)
    i = 0

    while i < n and len(result) < CANDIDATE_CAP:
        w = words[i]
        c = _strip_punctuation(w)

        is_candidate = (
            len(c) > 1
            and c[0].isupper()
            and not _is_stop_word(c)
        )

        if is_candidate:
            parts = [c]
            j = i + 1

            while j < n and not _ends_phrase(words[j - 1]) and len(parts) < 5:
                nc = _strip_punctuation(words[j])
                if len(nc) > 1 and nc[0].isupper() and not _is_stop_word(nc):
                    parts.append(nc)
                    j += 1
                else:
                    break

            phrase = " ".join(parts)
            if phrase not in seen:
                seen.add(phrase)
                result.append(phrase)
            i = j
        else:
            i += 1

    return result


def resolve_pronouns(
    text: str,
    entities: list[tuple[str, Optional[str]]],
    candidates: list[str],
) -> list[tuple[str, str]]:
    """Resolve pronouns in text to entity names.

    entities: list of (name, gender) from graph snapshot.
    candidates: Person-entity names only (e.g. GLiNER-confirmed spans).
    Returns list of (pronoun_lower, entity_name) pairs.
    """
    words = text.split()
    resolved: list[tuple[str, str]] = []
    seen_pronouns: set[str] = set()

    for idx, w in enumerate(words):
        lower = _strip_punctuation(w).lower()

        if lower in MALE_PRONOUNS:
            gender = "Male"
        elif lower in FEMALE_PRONOUNS:
            gender = "Female"
        elif lower in NEUTRAL_PRONOUNS:
            gender = "Neutral"
        else:
            continue

        if lower in seen_pronouns:
            continue
        seen_pronouns.add(lower)

        # Strategy 1: graph snapshot gender match
        neutral_candidates = [e for e in entities if e[1] is None]
        if gender == "Neutral" and len(neutral_candidates) != 1:
            continue

        found = next(
            (
                n for n, g in reversed(entities)
                if (g == gender if gender != "Neutral" else g is None)
            ),
            None,
        )

        name = found
        if name is None and gender != "Neutral":
            # Strategy 2: backward candidate scan
            name = _backward_candidate(words[:idx], candidates)
        if name is None:
            # Strategy 3: forward scan
            name = _forward_name(words[idx + 1:])

        if name:
            resolved.append((lower, name))

    return resolved


@dataclass
class CorefResolution:
    surface: str
    entity_name: str
    offset: int
    confidence: float
    method: str


DEFINITE_DESCRIPTIONS = [
    ("grandpa", "grandpa"),
    ("grandfather", "grandfather"),
    ("my grandfather", "my grandfather"),
    ("grandma", "grandma"),
    ("grandmother", "grandmother"),
    ("the author", "author"),
    ("the narrator", "narrator"),
    ("my mother", "mother"),
    ("his mother", "mother"),
    ("her mother", "mother"),
    ("my father", "father"),
    ("his father", "father"),
    ("her father", "father"),
    ("my wife", "wife"),
    ("his wife", "wife"),
    ("my husband", "husband"),
    ("her husband", "husband"),
]


def resolve_definite_descriptions(
    text: str,
    candidates: list[tuple[str, list[str], Optional[str]]],
) -> list[CorefResolution]:
    """Resolve definite descriptions and kinship roles to known entities by alias matching."""
    text_lower = text.lower()
    results = []
    for surface, alias_pattern in DEFINITE_DESCRIPTIONS:
        offset = text_lower.find(surface)
        if offset == -1:
            continue
        matched = next(
            (name for name, aliases, _ in candidates
             if any(a.lower() == alias_pattern for a in aliases)),
            None,
        )
        if matched:
            results.append(CorefResolution(
                surface=surface,
                entity_name=matched,
                offset=offset,
                confidence=0.9,
                method="alias_match",
            ))
    return results


def resolve_pronouns_from_candidates(
    text: str,
    candidates: list[tuple[str, list[str], Optional[str]]],
) -> list[CorefResolution]:
    """Resolve pronouns using gender matching against richer graph candidates."""
    words = text.split()
    results = []
    seen: set[str] = set()

    for idx, w in enumerate(words):
        lower = _strip_punctuation(w).lower()
        if lower in MALE_PRONOUNS:
            gender = "Male"
        elif lower in FEMALE_PRONOUNS:
            gender = "Female"
        elif lower in NEUTRAL_PRONOUNS:
            gender = "Neutral"
        else:
            continue
        if lower in seen:
            continue
        seen.add(lower)

        matched = next(
            (
                name for name, _, g in reversed(candidates)
                if ((g == gender or gender == "Neutral") if g is not None else gender == "Neutral")
            ),
            None,
        )
        if matched:
            offset = sum(len(ww) + 1 for ww in words[:idx + 1]) - len(w) - 1
            results.append(CorefResolution(
                surface=w,
                entity_name=matched,
                offset=max(0, offset),
                confidence=0.9,
                method="gender_nearest",
            ))
    return results


SPATIAL_PRONOUNS = {"there", "where"}

PLACE_DEFINITE_DESCRIPTIONS = [
    ("the district", "district"),
    ("the area", "area"),
    ("the neighbourhood", "neighbourhood"),
    ("the neighborhood", "neighborhood"),
    ("the suburb", "suburb"),
    ("the street", "street"),
    ("the road", "road"),
    ("the building", "building"),
    ("the mosque", "mosque"),
    ("the church", "church"),
    ("the field", "field"),
    ("the park", "park"),
]

ORG_DEFINITE_DESCRIPTIONS = [
    ("the organization", "organization"),
    ("the organisation", "organisation"),
    ("the group", "group"),
    ("the movement", "movement"),
    ("the committee", "committee"),
    ("the party", "party"),
    ("the league", "league"),
    ("the association", "association"),
    ("the college", "college"),
    ("the school", "school"),
    ("the congress", "congress"),
    ("the council", "council"),
]


def resolve_place_pronouns_from_candidates(
    text: str,
    in_chunk_candidates: list[tuple[str, list[str]]],
) -> list[CorefResolution]:
    if not in_chunk_candidates:
        return []

    words = text.split()
    text_lower = text.lower()
    results = []
    seen: set[str] = set()

    for idx, w in enumerate(words):
        lower = _strip_punctuation(w).lower()
        if lower not in SPATIAL_PRONOUNS or lower in seen:
            continue
        seen.add(lower)

        before_text = " ".join(words[:idx]).lower()
        best = max(
            in_chunk_candidates,
            key=lambda nc: max(
                (before_text.rfind(ww) for ww in nc[0].lower().split() if len(ww) >= 4),
                default=0,
            ),
        )
        name, _ = best
        nl = name.lower()
        mentioned = any(
            before_text.find(ww) != -1
            for ww in nl.split()
            if len(ww) >= 4
        )
        if mentioned:
            offset = sum(len(ww) + 1 for ww in words[:idx])
            results.append(CorefResolution(
                surface=w,
                entity_name=name,
                offset=offset,
                confidence=0.85,
                method="spatial_pronoun",
            ))

    for surface, alias_pat in PLACE_DEFINITE_DESCRIPTIONS:
        offset = text_lower.find(surface)
        if offset == -1:
            continue
        matched = next(
            (name for name, aliases in in_chunk_candidates
             if any(a.lower() == alias_pat for a in aliases)),
            None,
        )
        if matched:
            results.append(CorefResolution(
                surface=surface,
                entity_name=matched,
                offset=offset,
                confidence=0.9,
                method="place_alias_match",
            ))
    return results


def resolve_org_descriptions_from_candidates(
    text: str,
    candidates: list[tuple[str, list[str]]],
) -> list[CorefResolution]:
    if not candidates:
        return []
    text_lower = text.lower()
    results = []
    for surface, alias_pat in ORG_DEFINITE_DESCRIPTIONS:
        offset = text_lower.find(surface)
        if offset == -1:
            continue
        matched = next(
            (name for name, aliases in candidates
             if any(a.lower() == alias_pat for a in aliases)),
            None,
        )
        if matched:
            results.append(CorefResolution(
                surface=surface,
                entity_name=matched,
                offset=offset,
                confidence=0.9,
                method="org_alias_match",
            ))
    return results


def _backward_candidate(words_before: list[str], candidates: list[str]) -> Optional[str]:
    if not candidates or not words_before:
        return None
    before_text = " ".join(words_before).lower()
    best = max(
        (
            (pos, c)
            for c in candidates
            for pos in [before_text.rfind(c.lower())]
            if pos != -1
        ),
        key=lambda x: x[0],
        default=None,
    )
    return best[1] if best else None


def _forward_name(words: list[str]) -> Optional[str]:
    parts = []
    in_phrase = False
    for w in words[:40]:
        c = _strip_punctuation(w)
        is_candidate = len(c) > 1 and c[0].isupper() and not _is_stop_word(c)
        if is_candidate:
            parts.append(c)
            in_phrase = True
        elif in_phrase:
            break
    return " ".join(parts) if parts else None
