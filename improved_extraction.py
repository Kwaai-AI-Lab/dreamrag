"""
improved_extraction.py — High-quality entity and relation extraction.

Implements:
1. Better NER using GLiNER (preferred) + spaCy + pattern matching
2. Semantic relation extraction from text
3. Entity deduplication and merging
4. Confidence scoring
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from gliner_ner import (
    DEFAULT_THRESHOLD,
    DEFAULT_URL,
    extract_entities as gliner_extract,
    health_check,
    inprocess_available,
    is_valid_entity,
    normalize_entity_name,
)

logger = logging.getLogger(__name__)

# Try spaCy for better NER
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None
except ImportError:
    nlp = None


@dataclass
class ExtractedEntity:
    """Extracted entity with metadata."""
    name: str
    entity_type: str
    confidence: float
    source_tokens: List[str]


@dataclass
class ExtractedRelation:
    """Extracted relation between entities."""
    src_name: str
    src_type: str
    dst_name: str
    dst_type: str
    relation_type: str
    confidence: float
    evidence: str


class ImprovedEntityExtractor:
    """High-quality entity extraction combining multiple methods.

    Priority:
      1. GLiNER (HTTP server or in-process) — highest precision
      2. spaCy NER when available
      3. Pattern-based extraction (titles, org suffixes)
      4. Capitalization heuristics (only when GLiNER is unavailable)
    """

    # High-confidence person indicators
    PERSON_TITLES = {
        "mr", "mrs", "ms", "miss", "dr", "prof", "professor", "rev", "reverend",
        "sir", "lord", "lady", "captain", "general", "admiral", "president",
        "minister", "chancellor", "governor", "mayor", "judge", "justice",
    }

    PERSON_ROLES = {
        "author", "writer", "scientist", "professor", "doctor", "engineer",
        "physicist", "chemist", "biologist", "astronomer", "explorer",
        "inventor", "artist", "composer", "musician", "actor", "director",
        "politician", "diplomat", "general", "admiral", "president",
        "founder", "ceo", "director", "researcher", "mathematician",
    }

    # Organization patterns
    ORG_SUFFIXES = {
        "inc", "corp", "corporation", "ltd", "limited", "llc", "plc",
        "university", "college", "institute", "academy", "school",
        "hospital", "clinic", "agency", "department", "division",
        "foundation", "association", "society", "organization", "company",
    }

    # Relation patterns
    RELATION_PATTERNS = [
        # Person-to-Person
        (r"(\w+)\s+(?:married|wedded|spouse)\s+(\w+)", "spouse_of", 0.9),
        (r"(\w+)\s+(?:father|dad|papa)\s+(?:of|to)\s+(\w+)", "parent_of", 0.9),
        (r"(\w+)\s+(?:mother|mom|mama)\s+(?:of|to)\s+(\w+)", "parent_of", 0.9),
        (r"(\w+)\s+(?:son|daughter|child)\s+(?:of|to)\s+(\w+)", "child_of", 0.9),
        (r"(\w+)\s+(?:brother|sister|sibling)\s+of\s+(\w+)", "sibling_of", 0.9),
        (r"(\w+)\s+studied\s+under\s+(\w+)", "studied_under", 0.8),
        (r"(\w+)\s+mentored\s+(\w+)", "mentored", 0.8),
        (r"(\w+)\s+collaborated\s+with\s+(\w+)", "collaborated_with", 0.8),
        (r"(\w+)\s+worked\s+(?:with|for)\s+(\w+)", "worked_with", 0.7),

        # Person-to-Organization
        (r"(\w+)\s+(?:founded|established|created)\s+(\w+)", "founded", 0.9),
        (r"(\w+)\s+(?:directed|led|headed)\s+(\w+)", "led", 0.8),
        (r"(\w+)\s+worked\s+at\s+(\w+)", "worked_at", 0.8),
        (r"(\w+)\s+employed\s+by\s+(\w+)", "employed_by", 0.8),
        (r"(\w+)\s+professor\s+at\s+(\w+)", "worked_at", 0.9),

        # Location relations
        (r"(\w+)\s+located\s+in\s+(\w+)", "located_in", 0.8),
        (r"(\w+)\s+in\s+(?:the\s+)?(\w+)", "located_in", 0.6),
        (r"(\w+)\s+(?:capital|city)\s+of\s+(\w+)", "capital_of", 0.9),

        # Organization relations
        (r"(\w+)\s+part\s+of\s+(\w+)", "part_of", 0.8),
        (r"(\w+)\s+subsidiary\s+of\s+(\w+)", "subsidiary_of", 0.9),
        (r"(\w+)\s+division\s+of\s+(\w+)", "division_of", 0.9),

        # Concept relations
        (r"(\w+)\s+(?:discovered|invented)\s+(\w+)", "discovered", 0.8),
        (r"(\w+)\s+theorem", "named_theorem", 0.8),
        (r"(\w+)\s+law", "named_law", 0.8),
    ]

    def __init__(
        self,
        gliner_url: Optional[str] = DEFAULT_URL,
        gliner_threshold: float = DEFAULT_THRESHOLD,
        prefer_inprocess_gliner: bool = False,
        use_capitalization_fallback: bool = False,
        use_pattern_fallback: bool = True,
    ):
        self.spacy_available = nlp is not None
        self.gliner_url = gliner_url
        self.gliner_threshold = gliner_threshold
        self.prefer_inprocess_gliner = prefer_inprocess_gliner
        self.use_capitalization_fallback = use_capitalization_fallback
        self.use_pattern_fallback = use_pattern_fallback
        self._gliner_checked = False
        self._gliner_available = False

    @property
    def gliner_available(self) -> bool:
        """Lazily probe whether GLiNER can produce entities."""
        if not self._gliner_checked:
            self._gliner_checked = True
            if self.gliner_url and health_check(self.gliner_url):
                self._gliner_available = True
            else:
                self._gliner_available = inprocess_available()
        return self._gliner_available

    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract entities from text using multiple methods."""
        entities: Dict[tuple[str, str], ExtractedEntity] = {}

        # Method 1: GLiNER (preferred)
        gliner_entities = self._extract_gliner(text)
        for entity in gliner_entities:
            self._upsert(entities, entity)

        # Method 2: spaCy NER (if available) — still filtered
        if self.spacy_available:
            for entity in self._extract_spacy(text):
                self._upsert(entities, entity)

        # Method 3: Pattern-based extraction — only fill gaps when GLiNER is sparse
        if self.use_pattern_fallback and len(gliner_entities) < 2:
            for entity in self._extract_patterns(text):
                self._upsert(entities, entity)

        # Method 4: Capitalization heuristics — off by default (too noisy)
        if self.use_capitalization_fallback and not gliner_entities:
            for entity in self._extract_capitalization(text):
                self._upsert(entities, entity)

        return list(entities.values())

    @staticmethod
    def _upsert(
        entities: Dict[tuple[str, str], ExtractedEntity], entity: ExtractedEntity
    ) -> None:
        name = normalize_entity_name(entity.name)
        if not name:
            return
        entity = ExtractedEntity(
            name=name,
            entity_type=entity.entity_type,
            confidence=entity.confidence,
            source_tokens=name.split(),
        )
        if not is_valid_entity(entity.name, entity.entity_type, entity.confidence):
            return
        key = (entity.name.lower(), entity.entity_type)
        if key not in entities or entity.confidence > entities[key].confidence:
            entities[key] = entity

    def _extract_gliner(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using GLiNER (HTTP or in-process)."""
        if not text or not text.strip():
            return []

        chunks = _chunk_text(text, max_chars=3000)
        entities: List[ExtractedEntity] = []

        for chunk in chunks:
            spans = gliner_extract(
                chunk,
                url=self.gliner_url,
                threshold=self.gliner_threshold,
                prefer_inprocess=self.prefer_inprocess_gliner,
            )
            if spans:
                self._gliner_available = True
                self._gliner_checked = True

            for span in spans:
                entities.append(
                    ExtractedEntity(
                        name=span.text,
                        entity_type=span.entity_type,
                        confidence=float(span.score),
                        source_tokens=span.text.split(),
                    )
                )

        return entities

    def _extract_spacy(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using spaCy."""
        if not self.spacy_available:
            return []

        entities: List[ExtractedEntity] = []
        doc = nlp(text[:5000])  # Limit to avoid slowness

        for ent in doc.ents:
            entity_type = self._map_spacy_label(ent.label_)
            if entity_type:
                entities.append(
                    ExtractedEntity(
                        name=ent.text,
                        entity_type=entity_type,
                        confidence=0.85,  # spaCy confidence
                        source_tokens=[token.text for token in ent],
                    )
                )

        return entities

    def _extract_patterns(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using domain-specific patterns."""
        entities: List[ExtractedEntity] = []
        # (?-i:...) keeps capitalized-name groups case-sensitive even when
        # titles/suffixes are matched case-insensitively.
        name = r"((?-i:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*))"

        # Title-based person extraction
        for title in self.PERSON_TITLES:
            pattern = rf"(?i)\b{re.escape(title)}\.?\s+{name}\b"
            for match in re.finditer(pattern, text):
                entities.append(
                    ExtractedEntity(
                        name=match.group(1),
                        entity_type="Person",
                        confidence=0.85,
                        source_tokens=match.group(1).split(),
                    )
                )

        # Role-based person extraction
        for role in self.PERSON_ROLES:
            pattern = rf"{name}\s+(?i:(?:is\s+)?(?:an?\s+)?{re.escape(role)})\b"
            for match in re.finditer(pattern, text):
                entities.append(
                    ExtractedEntity(
                        name=match.group(1),
                        entity_type="Person",
                        confidence=0.75,
                        source_tokens=match.group(1).split(),
                    )
                )

        # Organization extraction
        for suffix in self.ORG_SUFFIXES:
            pattern = rf"\b{name}\s+(?i:{re.escape(suffix)})\b"
            for match in re.finditer(pattern, text):
                suffix_text = match.group(0)[len(match.group(1)) :].strip()
                full_name = f"{match.group(1)} {suffix_text}"
                entities.append(
                    ExtractedEntity(
                        name=full_name,
                        entity_type="Organization",
                        confidence=0.8,
                        source_tokens=full_name.split(),
                    )
                )

        return entities

    def _extract_capitalization(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using capitalization patterns."""
        entities: List[ExtractedEntity] = []
        name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'

        name_counts: Dict[str, int] = {}
        for match in re.finditer(name_pattern, text):
            name = match.group(1)
            if len(name) > 2:
                name_counts[name] = name_counts.get(name, 0) + 1

        # Keep names mentioned 2+ times
        for name, count in name_counts.items():
            if count >= 2 and not self._is_common_word(name):
                entities.append(
                    ExtractedEntity(
                        name=name,
                        entity_type="Person",  # Default to Person
                        confidence=min(0.6, 0.3 + 0.1 * count),  # Increase with frequency
                        source_tokens=name.split(),
                    )
                )

        return entities

    def extract_relations(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedRelation]:
        """Extract semantic relations from text."""
        relations: List[ExtractedRelation] = []
        entity_names = {e.name.lower(): (e.name, e.entity_type) for e in entities}

        for pattern, relation_type, confidence in self.RELATION_PATTERNS:
            try:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Handle patterns with more than 2 groups
                    if match.lastindex is None or match.lastindex < 2:
                        continue

                    src_name = match.group(1)
                    dst_name = match.group(2)

                    if not src_name or not dst_name:
                        continue

                    src_key = src_name.lower()
                    dst_key = dst_name.lower()

                    if src_key in entity_names and dst_key in entity_names:
                        src_display, src_type = entity_names[src_key]
                        dst_display, dst_type = entity_names[dst_key]

                        relations.append(
                            ExtractedRelation(
                                src_name=src_display,
                                src_type=src_type,
                                dst_name=dst_display,
                                dst_type=dst_type,
                                relation_type=relation_type,
                                confidence=confidence,
                                evidence=match.group(0),
                            )
                        )
            except (IndexError, AttributeError):
                # Skip patterns that don't match expected structure
                continue

        return relations

    @staticmethod
    def _map_spacy_label(label: str) -> Optional[str]:
        """Map spaCy label to our entity types."""
        mapping = {
            "PERSON": "Person",
            "ORG": "Organization",
            "GPE": "Location",
            "LOC": "Location",
            "PRODUCT": "Product",
            "EVENT": "Event",
            "FAC": "Facility",
            "WORK_OF_ART": "Work",
            "DATE": "Date",
        }
        return mapping.get(label)

    @staticmethod
    def _is_common_word(word: str) -> bool:
        """Check if word is too common to be an entity."""
        common = {
            "the", "and", "or", "but", "that", "this", "from", "with",
            "for", "are", "was", "were", "been", "being", "have", "has",
            "which", "who", "what", "when", "where", "why", "how",
        }
        return word.lower() in common


def deduplicate_entities(entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
    """Merge similar entities and return deduplicated list."""
    merged: Dict[str, ExtractedEntity] = {}

    # Prefer higher confidence, then longer more-specific names
    ordered = sorted(entities, key=lambda e: (-e.confidence, -len(e.name)))

    for entity in ordered:
        normalized = normalize_entity_name(entity.name).lower().strip()
        if not normalized:
            continue
        if not is_valid_entity(entity.name, entity.entity_type, entity.confidence):
            continue

        # Exact match
        if normalized in merged:
            if entity.confidence > merged[normalized].confidence:
                merged[normalized] = entity
            continue

        # Fuzzy / alias match (same last name, or title-stripped containment)
        found_key = None
        for key, existing in merged.items():
            if existing.entity_type != entity.entity_type:
                continue
            if _is_similar_name(normalized, key):
                found_key = key
                break
            # "professor ruth rasul" vs "ruth rasul"
            if normalized.endswith(key) or key.endswith(normalized):
                shorter, longer = sorted([normalized, key], key=len)
                if longer.endswith(shorter) and len(longer.split()) - len(shorter.split()) <= 2:
                    found_key = key
                    break

        if found_key is not None:
            existing = merged[found_key]
            # Keep the more specific (longer) name when confidence is close
            if len(entity.name) > len(existing.name) and entity.confidence >= existing.confidence - 0.1:
                del merged[found_key]
                merged[normalized] = entity
            elif entity.confidence > existing.confidence:
                merged[found_key] = entity
        else:
            merged[normalized] = entity

    return list(merged.values())


def _is_similar_name(name1: str, name2: str) -> bool:
    """Check if two names are similar."""
    # Last name match (for persons)
    parts1 = name1.split()
    parts2 = name2.split()

    if len(parts1) > 1 and len(parts2) > 1:
        if parts1[-1] == parts2[-1]:  # Same last name
            return True

    # Substring for org names
    if len(name1) > 10 and len(name2) > 10:
        words1 = set(parts1)
        words2 = set(parts2)
        if words1 and words2:
            overlap = len(words1 & words2) / max(len(words1), len(words2))
            if overlap >= 0.7:
                return True

    return False


def _chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    """Split text into roughly paragraph-sized chunks for NER."""
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    paragraphs = re.split(r"\n\s*\n", text)
    current = ""
    for para in paragraphs:
        if not para.strip():
            continue
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)

    # Hard-split any remaining oversized chunks
    final: List[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final.append(chunk)
            continue
        for i in range(0, len(chunk), max_chars):
            final.append(chunk[i : i + max_chars])
    return final or [text]


if __name__ == "__main__":
    sample = (
        "Marie Curie worked at the University of Paris and discovered radium "
        "while collaborating with Pierre Curie in France. She later joined "
        "the Radium Institute."
    )
    extractor = ImprovedEntityExtractor()
    print(
        f"GLiNER available={extractor.gliner_available} "
        f"spaCy available={extractor.spacy_available}"
    )
    for ent in extractor.extract_entities(sample):
        print(f"  {ent.name!r:35} {ent.entity_type:15} conf={ent.confidence:.3f}")
