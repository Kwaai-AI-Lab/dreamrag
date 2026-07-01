"""
corpus_schema.py — Per-corpus schema: which entity and relation types guide
graph construction for a given document collection.

The Dream RAG outline calls for defining a schema per corpus ("For each corpus,
specific entity types and relationship types are defined to guide graph
construction"). The hardcoded lists in graph.py (ENTITY_TYPES, RELATION_TYPES)
are the default/global schema; a CorpusSchema lets a specific corpus narrow or
extend them, and feeds directly into the ingestion graph config.

Usage:
    schema = CorpusSchema.load("schemas/manhattan_project.json")
    graph_cfg.entity_types = schema.entity_types        # ingestion already supports this
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from graph import ENTITY_TYPES as DEFAULT_ENTITY_TYPES
from graph import RELATION_TYPES as DEFAULT_RELATION_TYPES


@dataclass
class CorpusSchema:
    name: str = "default"
    description: str = ""
    entity_types: list[str] = field(default_factory=lambda: list(DEFAULT_ENTITY_TYPES))
    relation_types: list[str] = field(default_factory=lambda: list(DEFAULT_RELATION_TYPES))
    # Optional free-text guidance injected into extraction prompts / cluster summaries.
    domain_notes: str = ""

    @classmethod
    def default(cls) -> "CorpusSchema":
        return cls()

    @classmethod
    def load(cls, path: str | Path) -> "CorpusSchema":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=data.get("name", "custom"),
            description=data.get("description", ""),
            entity_types=data.get("entity_types") or list(DEFAULT_ENTITY_TYPES),
            relation_types=data.get("relation_types") or list(DEFAULT_RELATION_TYPES),
            domain_notes=data.get("domain_notes", ""),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "name": self.name,
                    "description": self.description,
                    "entity_types": self.entity_types,
                    "relation_types": self.relation_types,
                    "domain_notes": self.domain_notes,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
