"""
graph.py — Knowledge graph: entity nodes, directed relations, and LLM-based extraction.

Mirrors graph.rs. Persists to a per-tenant SQLite database.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import sqlite3
import struct
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator, Optional
from uuid import UUID

import aiohttp

logger = logging.getLogger(__name__)

# ── Ontology constants ────────────────────────────────────────────────────────

ENTITY_TYPES = [
    "Person", "Organization", "Location", "Event", "Concept", "Method",
    "Claim", "Quantity", "Date", "Document", "Product", "Technology",
    "Role", "Topic", "Unknown",
]

RELATION_TYPES = [
    "parent_of", "child_of", "spouse_of", "sibling_of", "half_sibling_of",
    "grandparent_of", "grandchild_of", "uncle_of", "aunt_of", "niece_of",
    "nephew_of", "cousin_of", "foster_parent_of", "foster_child_of",
    "works_at", "founded", "manages", "belongs_to", "member_of", "led", "endorses",
    "lived_in", "visited", "built",
    "part_of", "contains", "located_in", "instance_of", "subtype_of",
    "occurred_on", "started", "ended", "followed_by", "precedes",
    "related_to", "contradicts", "supports", "cites", "implements",
    "defined_by", "described_in", "measured_by", "associated_with", "caused_by",
]

PERSON_RELATION_TYPES = [
    "parent_of", "child_of", "spouse_of", "sibling_of", "half_sibling_of",
    "grandparent_of", "grandchild_of", "uncle_of", "aunt_of", "niece_of",
    "nephew_of", "cousin_of", "foster_parent_of", "foster_child_of",
    "works_at", "belongs_to", "endorses", "associated_with", "related_to", "supported",
]

FAMILIAL_RELS = {
    "parent_of", "child_of", "spouse_of", "sibling_of", "half_sibling_of",
    "grandparent_of", "grandchild_of", "uncle_of", "aunt_of", "niece_of",
    "nephew_of", "cousin_of", "foster_parent_of", "foster_child_of",
}

FAMILIAL_INVERSE = {
    "parent_of": "child_of",
    "child_of": "parent_of",
    "grandparent_of": "grandchild_of",
    "grandchild_of": "grandparent_of",
    "uncle_of": "nephew_of",
    "aunt_of": "niece_of",
    "nephew_of": "uncle_of",
    "niece_of": "aunt_of",
    "foster_parent_of": "foster_child_of",
    "foster_child_of": "foster_parent_of",
}

SYMMETRIC_FAMILIAL = {"spouse_of", "sibling_of", "half_sibling_of", "cousin_of"}

# ── Core types ────────────────────────────────────────────────────────────────

@dataclass
class FieldValue:
    value: str
    evidence_chunk_ids: list[int] = field(default_factory=list)
    confidence: float = 1.0 / 3.0

    @classmethod
    def new(cls, value: str, chunk_id: int) -> "FieldValue":
        return cls(value=value, evidence_chunk_ids=[chunk_id], confidence=1.0 / 3.0)

    def add_evidence(self, chunk_id: int) -> None:
        if chunk_id not in self.evidence_chunk_ids:
            self.evidence_chunk_ids.append(chunk_id)
        self.confidence = min(1.0, len(self.evidence_chunk_ids) / 3.0)


@dataclass
class EntityNode:
    id: int
    name: str
    entity_type: str
    description: str
    embedding: list[float]
    mention_count: int = 1
    first_chunk_id: int = 0
    aliases: list[str] = field(default_factory=list)
    schema_type: Optional[str] = None
    gender: Optional[str] = None
    evidence: list[int] = field(default_factory=list)
    fields: dict[str, FieldValue] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class RelationRecord:
    src_id: int
    dst_id: int
    relation_type: str
    strength: float = 0.0
    evidence_chunk_ids: list[int] = field(default_factory=list)


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str
    description: str = ""
    fields: dict[str, str] = field(default_factory=dict)


@dataclass
class ExtractedRelation:
    from_: str
    to: str
    relation: str


# ── Deterministic entity ID ───────────────────────────────────────────────────

def entity_id(name: str, entity_type: str) -> int:
    h = hashlib.sha256()
    h.update(name.lower().encode())
    h.update(b"::")
    h.update(entity_type.encode())
    digest = h.digest()
    return struct.unpack_from("<q", digest[:8])[0]


# ── Schema field registry ─────────────────────────────────────────────────────

#this is going to change a lot, generalized each document type will have its unique schema and unique fields


_EXPECTED_FIELDS: dict[str, list[tuple[str, str]]] = {
    "Person": [
        ("birthDate", "date of birth"),
        ("birthPlace", "place of birth"),
        ("deathDate", "date of death (if deceased)"),
        ("nationality", "nationality or cultural identity"),
        ("occupation", "profession or main occupation"),
        ("affiliation", "organization they belong or belonged to"),
        ("spouse", "spouse or partner name"),
        ("parent", "parent names"),
        ("sibling", "sibling names"),
        ("child", "child names"),
    ],
    "Place": [
        ("addressLocality", "city, district or suburb"),
        ("addressRegion", "province or region"),
        ("addressCountry", "country"),
        ("locationType", "type of place (district, city, country, neighbourhood)"),
        ("historicalNote", "historical significance or period"),
    ],
    "Location": [
        ("addressLocality", "city, district or suburb"),
        ("addressRegion", "province or region"),
        ("addressCountry", "country"),
        ("locationType", "type of place (district, city, country, neighbourhood)"),
        ("historicalNote", "historical significance or period"),
    ],
    "Organization": [
        ("foundingDate", "year or period when founded"),
        ("dissolutionDate", "year or period when dissolved, if applicable"),
        ("location", "city or country of headquarters or main office"),
        ("founder", "founder name"),
        ("orgType", "type of organization (school, mosque, political party, etc.)"),
    ],
}

_PLACEHOLDER_VALUES = {
    "unknown", "undefined", "n/a", "none", "not stated", "not specified",
    "not mentioned", "not known", "not applicable", "unspecified",
}


def expected_fields(entity_type: str) -> list[tuple[str, str]]:
    return _EXPECTED_FIELDS.get(entity_type, [])


def description_from_fields(
    name: str,
    entity_type: str,
    fields: dict[str, Any],
) -> str:
    schema = expected_fields(entity_type)
    if not schema or not fields:
        return ""

    def _value(fv: Any) -> Optional[str]:
        if isinstance(fv, FieldValue):
            v = fv.value
        elif isinstance(fv, dict):
            v = fv.get("value", "")
        else:
            v = str(fv)
        if not v or v.lower().strip() in _PLACEHOLDER_VALUES:
            return None
        return v

    parts = [
        f"{key}: {v}"
        for key, _ in schema
        if (v := _value(fields.get(key))) is not None
    ]
    return f"{name} — {'; '.join(parts)}" if parts else ""


# ── Name normalization helpers ────────────────────────────────────────────────

def normalize_name(s: str) -> str:
    return " ".join(
        "".join(c if c.isalnum() else " " for c in s).split()
    ).lower()


_HONORIFICS = {
    "dr", "mr", "mrs", "ms", "miss", "prof", "professor", "rev", "reverend",
    "sir", "haji", "hajj", "maulvi", "maulana", "imam", "sheikh", "shaykh",
    "auntie", "aunt", "uncle", "oom", "tannie", "oupa", "my",
}

_QUALS = {
    "ma", "ba", "bsc", "msc", "phd", "llb", "llm", "bed", "bcom", "mba",
    "mpa", "hons", "dip", "jp", "obe", "mbe", "mbbs",
}


def _is_qual(tok: str) -> bool:
    undotted = tok.replace(".", "").lower()
    return bool(undotted) and undotted in _QUALS


def _strip_qualifications(name: str) -> Optional[str]:
    tokens = name.split()
    end = len(tokens)
    while end > 0 and _is_qual(tokens[end - 1]):
        end -= 1
    if end == len(tokens):
        return None
    return normalize_name(" ".join(tokens[:end]))


def _stripped_key(name: str) -> str:
    norm = normalize_name(name)
    return " ".join(w for w in norm.split() if w not in _HONORIFICS)


def _edit_distance(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(la):
        curr = [i + 1] + [0] * lb
        for j in range(lb):
            cost = 0 if a[i] == b[j] else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    return prev[lb]


def ord_pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _family_role_contradicts(r1: str, r2: str) -> bool:
    if r1 == r2:
        return False
    pairs = {
        ("spouse_of", "sibling_of"), ("spouse_of", "half_sibling_of"),
        ("spouse_of", "child_of"), ("spouse_of", "parent_of"),
        ("spouse_of", "grandparent_of"), ("spouse_of", "grandchild_of"),
        ("parent_of", "child_of"), ("grandparent_of", "grandchild_of"),
        ("sibling_of", "parent_of"), ("sibling_of", "child_of"),
        ("half_sibling_of", "parent_of"), ("half_sibling_of", "child_of"),
    }
    return (r1, r2) in pairs or (r2, r1) in pairs


# ── GraphStore ────────────────────────────────────────────────────────────────

class GraphStore:
    """In-memory knowledge graph backed by SQLite for persistence."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._con = sqlite3.connect(db_path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self.lock = asyncio.Lock()
        self._nodes: dict[int, EntityNode] = {}
        self._adj: dict[int, list[tuple[int, str, float]]] = {}
        self._chunk_to_entities: dict[int, list[int]] = {}
        self._entity_to_chunks: dict[int, list[int]] = {}
        self.alias_token_index: dict[str, list[int]] = {}
        self._init_tables()
        self._rebuild()

    @classmethod
    def open(cls, data_dir: str | Path, tenant_id: UUID) -> "GraphStore":
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(data_dir / f"graph-{tenant_id}.db")
        return cls(db_path)

    def _init_tables(self) -> None:
        self._con.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id INTEGER PRIMARY KEY,
                node_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS relations (
                src_id INTEGER,
                dst_id INTEGER,
                relation_type TEXT,
                rel_json TEXT NOT NULL,
                PRIMARY KEY (src_id, dst_id, relation_type)
            );
            CREATE TABLE IF NOT EXISTS chunk_entities (
                chunk_id INTEGER PRIMARY KEY,
                entity_ids TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entity_chunks (
                entity_id INTEGER PRIMARY KEY,
                chunk_ids TEXT NOT NULL
            );
        """)
        self._con.commit()

    def _node_to_dict(self, node: EntityNode) -> dict:
        d = asdict(node)
        d["fields"] = {
            k: asdict(v) if isinstance(v, FieldValue) else v
            for k, v in node.fields.items()
        }
        return d

    def _dict_to_node(self, d: dict) -> EntityNode:
        fields = {
            k: FieldValue(**v) if isinstance(v, dict) else v
            for k, v in d.get("fields", {}).items()
        }
        d = dict(d)
        d["fields"] = fields
        d.pop("evidence", None)
        return EntityNode(**d)

    def _rebuild(self) -> None:
        for row in self._con.execute("SELECT node_json FROM entities"):
            node = self._dict_to_node(json.loads(row["node_json"]))
            self._nodes[node.id] = node

        for row in self._con.execute("SELECT rel_json FROM relations"):
            rel = RelationRecord(**json.loads(row["rel_json"]))
            self._adj.setdefault(rel.src_id, []).append((rel.dst_id, rel.relation_type, rel.strength))
            self._adj.setdefault(rel.dst_id, []).append((rel.src_id, rel.relation_type, rel.strength))

        for row in self._con.execute("SELECT chunk_id, entity_ids FROM chunk_entities"):
            self._chunk_to_entities[row["chunk_id"]] = json.loads(row["entity_ids"])

        for row in self._con.execute("SELECT entity_id, chunk_ids FROM entity_chunks"):
            self._entity_to_chunks[row["entity_id"]] = json.loads(row["chunk_ids"])

        for eid, cids in self._entity_to_chunks.items():
            if eid in self._nodes:
                self._nodes[eid].evidence = cids

        self._rebuild_alias_index()
        logger.info(
            "graph store loaded entities=%d relations=%d",
            len(self._nodes),
            sum(len(v) for v in self._adj.values()) // 2,
        )

    def _rebuild_alias_index(self) -> None:
        self.alias_token_index.clear()
        for eid, node in self._nodes.items():
            forms = [node.name] + node.aliases
            for form in forms:
                for token in form.split():
                    raw = token.lower()
                    trimmed = raw.strip("".join(c for c in raw if not c.isalnum()))
                    for tok in {raw, trimmed}:
                        if len(tok) >= 2:
                            self.alias_token_index.setdefault(tok, []).append(eid)

    # ── Writes ────────────────────────────────────────────────────────────────

    def upsert_entity(self, node: EntityNode) -> None:
        existing = self._nodes.get(node.id)
        if existing:
            merged_fields = dict(existing.fields)
            for key, new_fv in node.fields.items():
                if key in merged_fields:
                    efv = merged_fields[key]
                    for cid in new_fv.evidence_chunk_ids:
                        efv.add_evidence(cid)
                    if not efv.value and new_fv.value:
                        efv.value = new_fv.value
                else:
                    merged_fields[key] = FieldValue(
                        value=new_fv.value if isinstance(new_fv, FieldValue) else new_fv.get("value", ""),
                        evidence_chunk_ids=list(new_fv.evidence_chunk_ids) if isinstance(new_fv, FieldValue) else new_fv.get("evidence_chunk_ids", []),
                        confidence=new_fv.confidence if isinstance(new_fv, FieldValue) else new_fv.get("confidence", 1/3),
                    )

            computed = description_from_fields(existing.name, existing.entity_type, merged_fields)
            best_desc = max(
                filter(None, [computed, node.description, existing.description]),
                key=len,
                default="",
            )
            best_emb = node.embedding if best_desc == node.description else existing.embedding
            merged_aliases = list(existing.aliases)
            for a in node.aliases:
                if a not in merged_aliases:
                    merged_aliases.append(a)

            merged = EntityNode(
                id=node.id,
                name=existing.name,
                entity_type=existing.entity_type,
                description=best_desc,
                embedding=best_emb,
                mention_count=existing.mention_count + 1,
                first_chunk_id=existing.first_chunk_id,
                aliases=merged_aliases,
                schema_type=existing.schema_type or node.schema_type,
                gender=existing.gender or node.gender,
                evidence=existing.evidence,
                fields=merged_fields,
                confidence=0.0,
            )
        else:
            merged = node

        node_dict = self._node_to_dict(merged)
        node_dict.pop("evidence", None)
        with self._con:
            self._con.execute(
                "INSERT OR REPLACE INTO entities(entity_id, node_json) VALUES (?,?)",
                (merged.id, json.dumps(node_dict)),
            )
        self._nodes[merged.id] = merged

    def upsert_relation(
        self,
        src_id: int,
        dst_id: int,
        relation_type: str,
        evidence_chunk_id: int,
    ) -> None:
        if relation_type in ("located_in", "works_at"):
            dst_node = self._nodes.get(dst_id)
            if dst_node and dst_node.schema_type == "schema:CreativeWork":
                return

        if relation_type in FAMILIAL_RELS:
            src_is_person = self._nodes.get(src_id, None)
            dst_is_person = self._nodes.get(dst_id, None)
            if src_is_person and src_is_person.entity_type.lower() != "person":
                return
            if dst_is_person and dst_is_person.entity_type.lower() != "person":
                return

        self._upsert_relation_unchecked(src_id, dst_id, relation_type, evidence_chunk_id)

        if inverse := FAMILIAL_INVERSE.get(relation_type):
            self._upsert_relation_unchecked(dst_id, src_id, inverse, evidence_chunk_id)

        if relation_type in SYMMETRIC_FAMILIAL:
            self._upsert_relation_unchecked(dst_id, src_id, relation_type, evidence_chunk_id)

    def _upsert_relation_unchecked(
        self, src_id: int, dst_id: int, relation_type: str, chunk_id: int
    ) -> None:
        row = self._con.execute(
            "SELECT rel_json FROM relations WHERE src_id=? AND dst_id=? AND relation_type=?",
            (src_id, dst_id, relation_type),
        ).fetchone()

        if row:
            rel = RelationRecord(**json.loads(row["rel_json"]))
            if chunk_id not in rel.evidence_chunk_ids:
                rel.evidence_chunk_ids.append(chunk_id)
            rel.strength = min(1.0, len(rel.evidence_chunk_ids) / 10.0)
        else:
            rel = RelationRecord(
                src_id=src_id,
                dst_id=dst_id,
                relation_type=relation_type,
                strength=0.1,
                evidence_chunk_ids=[chunk_id],
            )

        with self._con:
            self._con.execute(
                "INSERT OR REPLACE INTO relations(src_id, dst_id, relation_type, rel_json) VALUES (?,?,?,?)",
                (src_id, dst_id, relation_type, json.dumps(asdict(rel))),
            )

        edges = self._adj.setdefault(src_id, [])
        for i, (d, rt, _) in enumerate(edges):
            if d == dst_id and rt == relation_type:
                edges[i] = (dst_id, relation_type, rel.strength)
                break
        else:
            edges.append((dst_id, relation_type, rel.strength))

        edges_rev = self._adj.setdefault(dst_id, [])
        for i, (s, rt, _) in enumerate(edges_rev):
            if s == src_id and rt == relation_type:
                edges_rev[i] = (src_id, relation_type, rel.strength)
                break
        else:
            edges_rev.append((src_id, relation_type, rel.strength))

    def link_chunk(self, chunk_id: int, entity_ids: list[int]) -> None:
        if not entity_ids:
            return
        existing_c = self._chunk_to_entities.get(chunk_id, [])
        merged_c = sorted(set(existing_c) | set(entity_ids))
        with self._con:
            self._con.execute(
                "INSERT OR REPLACE INTO chunk_entities(chunk_id, entity_ids) VALUES (?,?)",
                (chunk_id, json.dumps(merged_c)),
            )
        self._chunk_to_entities[chunk_id] = merged_c

        for eid in entity_ids:
            existing_e = self._entity_to_chunks.get(eid, [])
            if chunk_id not in existing_e:
                merged_e = sorted(set(existing_e) | {chunk_id})
                with self._con:
                    self._con.execute(
                        "INSERT OR REPLACE INTO entity_chunks(entity_id, chunk_ids) VALUES (?,?)",
                        (eid, json.dumps(merged_e)),
                    )
                self._entity_to_chunks[eid] = merged_e
                if eid in self._nodes:
                    self._nodes[eid].evidence = merged_e

    def sync_evidence(self) -> None:
        """Populate entity.evidence from the chunk index."""
        for eid, cids in self._entity_to_chunks.items():
            if eid in self._nodes:
                self._nodes[eid].evidence = list(cids)

    # ── Reads ─────────────────────────────────────────────────────────────────

    def all_entities(self) -> Iterator[EntityNode]:
        return iter(self._nodes.values())

    def node_count(self) -> int:
        return len(self._nodes)

    def relation_count(self) -> int:
        return sum(len(v) for v in self._adj.values()) // 2

    def find_by_name(self, name: str) -> Optional[EntityNode]:
        norm = normalize_name(name)
        for node in self._nodes.values():
            if normalize_name(node.name) == norm:
                return node
            for alias in node.aliases:
                if normalize_name(alias) == norm:
                    return node
        return None

    def get_entity(self, entity_id: int) -> Optional[EntityNode]:
        return self._nodes.get(entity_id)

    def get_relations(self, entity_id: int) -> list[tuple[int, str, float]]:
        return list(self._adj.get(entity_id, []))

    # ── Confidence scoring ────────────────────────────────────────────────────

    def score_all_confidences(self) -> None:
        for node in self._nodes.values():
            node.confidence = self._score_entity(node)
        with self._con:
            for node in self._nodes.values():
                row = self._con.execute(
                    "SELECT node_json FROM entities WHERE entity_id=?", (node.id,)
                ).fetchone()
                if row:
                    d = json.loads(row["node_json"])
                    d["confidence"] = node.confidence
                    self._con.execute(
                        "INSERT OR REPLACE INTO entities(entity_id, node_json) VALUES (?,?)",
                        (node.id, json.dumps(d)),
                    )

    def _score_entity(self, node: EntityNode) -> float:
        schema = expected_fields(node.entity_type)
        if not schema:
            return min(1.0, node.mention_count / 3.0)
        filled = sum(
            1 for key, _ in schema
            if key in node.fields and node.fields[key].value
            and node.fields[key].value.lower().strip() not in _PLACEHOLDER_VALUES
        )
        return filled / len(schema) if schema else 0.0

    def rescore_entity(self, eid: int) -> float:
        node = self._nodes.get(eid)
        if not node:
            return 0.0
        score = self._score_entity(node)
        node.confidence = score
        row = self._con.execute(
            "SELECT node_json FROM entities WHERE entity_id=?", (eid,)
        ).fetchone()
        if row:
            d = json.loads(row["node_json"])
            d["confidence"] = score
            with self._con:
                self._con.execute(
                    "INSERT OR REPLACE INTO entities(entity_id, node_json) VALUES (?,?)",
                    (eid, json.dumps(d)),
                )
        return score

    def close(self) -> None:
        self._con.close()


# ── Entity name cleaning ──────────────────────────────────────────────────────

def clean_entity_name(name: str) -> str:
    """Fix PDF-extraction underscore artifacts in entity names."""
    # Normalise typographic quotes
    name = "".join(
        "'" if c in "\u2018\u2019\u201a\u201b" else c
        for c in name
    )

    # _Word_ → (Word) parenthetical patterns
    while True:
        m = re.search(r'(?:^| )_([^_]+)_(?= |$)', name)
        if not m:
            break
        name = name[:m.start()] + (" " if m.start() > 0 else "") + f"({m.group(1)})" + name[m.end():]

    # Per-character underscore rules
    chars = list(name)
    n = len(chars)
    out = []
    i = 0
    while i < n:
        c = chars[i]
        if c != "_":
            out.append(c)
            i += 1
            continue

        # Rule 1: _s → 's
        if i + 1 < n and chars[i + 1] == "s":
            after = i + 2
            if after >= n or not chars[after].isalpha() or chars[after].isupper():
                out.extend(["'", "s"])
                i += 2
                continue

        # Rule 2: _ preceded by alpha and followed by space/end/uppercase → .
        prev_alpha = bool(out) and out[-1].isalpha()
        next_ch = chars[i + 1] if i + 1 < n else None
        next_break = next_ch is None or next_ch == " " or (next_ch.isalpha() and next_ch.isupper())
        if prev_alpha and next_break:
            out.append(".")
        i += 1

    return "".join(out).strip()


# ── LLM entity extraction ─────────────────────────────────────────────────────

async def extract_from_text(
    text: str,
    candidates: list[str],
    pronoun_map: list[tuple[str, str]],
    section_note: Optional[str],
    inference_url: str,
    model: str,
    entity_types: list[str],
    no_relations: bool,
    gliner_hints: Optional[list[str]] = None,
) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    """Call the LLM to extract entities and relations from text.

    Returns (entities, relations). All failures degrade to ([], []).
    """
    if not candidates:
        logger.debug("no proper noun candidates — skipping LLM extraction")
        return [], []

    effective_types = entity_types if entity_types else ENTITY_TYPES
    entity_list = ", ".join(effective_types)

    section_context = f"DOCUMENT CONTEXT: {section_note}\n\n" if section_note else ""

    pronoun_context = ""
    if pronoun_map:
        pairs = ", ".join(f"'{p}' = '{n}'" for p, n in pronoun_map)
        pronoun_context = f"KNOWN COREFERENCES: {pairs}\n\n"

    entity_cap = 25 if len(entity_types) <= 3 else 20

    normalized_candidates = [clean_entity_name(c) for c in candidates]
    candidates_block = "\n".join(f"- {c}" for c in normalized_candidates)

    hints_block = ""
    if gliner_hints:
        hint_list = "\n".join(f"- {h}" for h in gliner_hints)
        hints_block = (
            "CONFIRMED PERSON NAMES (detected by a dedicated NER model — treat these as "
            f"high-confidence Person entities if they appear in the text):\n{hint_list}\n\n"
        )

    if no_relations:
        prompt = (
            f"{section_context}"
            f"{pronoun_context}"
            "You are a precise knowledge extraction engine.\n"
            "The following proper noun candidates were identified in the text.\n"
            "Classify each as a named entity (keep) or discard it if it is not a real entity.\n"
            "For kept entities output: name, type, and structured fields.\n"
            f"List AT MOST {entity_cap} entities.\n"
            'Return ONLY valid JSON (no markdown, no explanation):\n'
            '{"entities":[{"name":"...","type":"...","fields":{...}},...]}' "\n\n"
            f"{hints_block}"
            f"Candidates:\n{candidates_block}\n\n"
            f"Entity types: {entity_list}\n\n"
            "Field keys by entity type — include only keys whose values appear in the text:\n"
            "  Person:       birthDate, birthPlace, deathDate, nationality, occupation, "
            "affiliation, spouse, parent, sibling, child\n"
            "  Place:        addressLocality, addressRegion, addressCountry, locationType, historicalNote\n"
            "  Organization: foundingDate, dissolutionDate, location, founder, orgType\n\n"
            "IMPORTANT RULES:\n"
            "- Never create an entity whose name is a pronoun or generic role.\n"
            "- Only keep candidates that are real proper names, organisations, or places.\n"
            "- Entity names must be ≤ 5 words.\n"
            "- Omit any field whose value is not clearly stated in the text.\n"
            "- NEVER extract generic family roles as entity names.\n"
            "- Do NOT extract ethnic or racial group nouns as Person entities.\n"
            "- Do NOT extract ideological or political labels.\n"
            'If no candidates are real entities, return {"entities":[]}.\n\n'
            f"Text:\n{text}"
        )
    else:
        person_only = len(entity_types) == 1 and entity_types[0].lower() == "person"
        relation_list = ", ".join(PERSON_RELATION_TYPES if person_only else RELATION_TYPES)
        prompt = (
            f"{section_context}"
            f"{pronoun_context}"
            "You are a precise knowledge extraction engine.\n"
            "The following proper noun candidates were identified in the text.\n"
            "Classify each as a named entity (keep) or discard it if it is not a real entity,\n"
            "then extract relationships between kept entities.\n"
            f"List AT MOST {entity_cap} entities.\n"
            'Return ONLY valid JSON (no markdown, no explanation):\n'
            '{"entities":[{"name":"...","type":"...","description":"1-2 sentences"},...], '
            '"relations":[{"from":"entity name","to":"entity name","relation":"relation_type"},...]}'
            "\n\n"
            f"{hints_block}"
            f"Candidates:\n{candidates_block}\n\n"
            f"Entity types: {entity_list}\n"
            f"Relation types: {relation_list}\n\n"
            "IMPORTANT RULES:\n"
            "- Never create an entity whose name is a pronoun or generic role.\n"
            "- Only keep candidates that are real proper names, organisations, or places.\n"
            "- Entity names must be ≤ 5 words.\n"
            "- NEVER extract generic family roles as entity names.\n"
            "- Do NOT extract ethnic or racial group nouns as Person entities.\n"
            "- Do NOT extract ideological or political labels.\n"
            "- Only assert a relation when the text EXPLICITLY STATES IT.\n"
            '- If no candidates are real entities, return {"entities":[],"relations":[]}.\n\n'
            f"Text:\n{text}"
        )

    url = f"{inference_url.rstrip('/')}/api/chat"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {"temperature": 0.1, "num_predict": 1024, "num_ctx": 8192},
    }

    timeout = aiohttp.ClientTimeout(total=120, connect=10)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=body) as resp:
                if not resp.ok:
                    logger.warning("entity extraction got HTTP %s", resp.status)
                    return [], []
                raw_text = await resp.text()
    except asyncio.TimeoutError:
        logger.warning("entity extraction timed out after 120s")
        return [], []
    except Exception as e:
        logger.warning("entity extraction request failed: %s", e)
        return [], []

    # Accumulate streaming NDJSON content tokens
    content_buf = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            v = json.loads(line)
            if c := v.get("message", {}).get("content"):
                content_buf.append(c)
        except json.JSONDecodeError:
            pass

    content = "".join(content_buf).strip()
    cleaned = content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.debug("entity extraction JSON parse failed: %s; raw: %.200s", e, cleaned)
        return [], []

    entities = []
    for item in payload.get("entities", []):
        name = clean_entity_name(item.get("name", ""))
        entities.append(ExtractedEntity(
            name=name,
            entity_type=item.get("type", "Unknown"),
            description=item.get("description", ""),
            fields=item.get("fields", {}),
        ))

    relations: list[ExtractedRelation] = []
    if not no_relations:
        for item in payload.get("relations", []):
            relations.append(ExtractedRelation(
                from_=clean_entity_name(item.get("from", "")),
                to=clean_entity_name(item.get("to", "")),
                relation=item.get("relation", ""),
            ))

    return entities, relations
