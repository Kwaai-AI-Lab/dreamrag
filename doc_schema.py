"""
doc_schema.py — Document schema definitions, section matching, and auto-detection.

Mirrors doc_schema.rs. Uses dataclasses and YAML loading (pyyaml).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


class SectionType(str, Enum):
    """Semantic type of a document section."""
    Main = "main"
    FrontMatter = "front_matter"
    TableOfContents = "table_of_contents"
    Preface = "preface"
    Introduction = "introduction"
    Chapter = "chapter"
    Acknowledgements = "acknowledgements"
    EditorNote = "editor_note"
    Appendix = "appendix"
    Bibliography = "bibliography"
    Index = "index"
    EndNotes = "end_notes"
    Caption = "caption"

    def same_window_zone(self, other: "SectionType") -> bool:
        """True when two section types belong to the same context-window zone.

        Windows may NOT span a zone boundary. Narrative sections (Main, Chapter,
        Introduction, Preface) are in the same zone; all others are isolated.
        """
        narrative = {SectionType.Main, SectionType.Chapter, SectionType.Introduction, SectionType.Preface}
        return self in narrative and other in narrative


@dataclass
class SectionDef:
    pattern: str
    skip: bool = False
    narrator_note: Optional[str] = None
    index_seeds: bool = False
    section_type: SectionType = SectionType.Main


@dataclass
class DocSchema:
    """A document schema controlling section tagging, skip flags, and narrator overrides."""
    document_title: Optional[str] = None
    default_narrator: Optional[str] = None
    sections: list[SectionDef] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    schema_type: Optional[str] = None

    def context_line(self) -> Optional[str]:
        """One-line context string injected into LLM prompts."""
        if not self.metadata and not self.document_title:
            return None
        parts = []
        title = (
            self.document_title
            or self.metadata.get("title")
            or "(untitled document)"
        )
        if author := self.metadata.get("author"):
            parts.append(f'"{title}" by {author}')
        else:
            parts.append(f'"{title}"')
        if year := (self.metadata.get("year") or self.metadata.get("datePublished")):
            parts.append(f"({year})")
        if isbn := self.metadata.get("isbn"):
            parts.append(f"ISBN: {isbn}")
        if pub := self.metadata.get("publisher"):
            parts.append(f"Publisher: {pub}")
        if subj := (self.metadata.get("subject") or self.metadata.get("about")):
            parts.append(f"Subject: {subj}")
        return " ".join(parts)

    def has_index_seeds(self) -> bool:
        return any(s.index_seeds for s in self.sections)


def load_doc_schema(path: str | Path) -> DocSchema:
    if yaml is None:
        raise RuntimeError("PyYAML is required for doc schema loading: pip install pyyaml")
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)

    sections = [
        SectionDef(
            pattern=s["pattern"],
            skip=s.get("skip", False),
            narrator_note=s.get("narrator_note"),
            index_seeds=s.get("index_seeds", False),
            section_type=SectionType(s.get("section_type", "main")),
        )
        for s in data.get("sections", [])
    ]

    return DocSchema(
        document_title=data.get("document_title"),
        default_narrator=data.get("default_narrator"),
        sections=sections,
        metadata=data.get("metadata", {}),
        schema_type=data.get("schema_type"),
    )


def match_section(heading: str, schema: DocSchema) -> Optional[SectionDef]:
    """Return the first SectionDef whose pattern is a case-insensitive substring of heading."""
    lower = heading.lower()
    for sec in schema.sections:
        if sec.pattern.lower() in lower:
            return sec
    return None


def auto_detect_schema(header_text: str) -> DocSchema:
    """Heuristic auto-detection of document type and metadata from the first ~3000 chars."""
    schema = DocSchema()
    if isbn := _find_isbn(header_text):
        schema.schema_type = "Book"
        schema.metadata["isbn"] = isbn
    if pub := _find_publisher(header_text):
        schema.metadata["publisher"] = pub
    if year := _find_copyright_year(header_text):
        schema.metadata.setdefault("copyrightYear", year)
        schema.metadata.setdefault("datePublished", year)
        schema.metadata.setdefault("year", year)
    if holder := _find_copyright_holder(header_text):
        schema.metadata["copyrightHolder"] = holder
    if author := _find_by_author(header_text):
        schema.metadata.setdefault("author", author)
    return schema


def parse_index_seeds(index_text: str) -> list[tuple[str, Optional[str]]]:
    """Parse an index section's raw text into (entity_name, type_hint) pairs."""
    seeds = []
    for line in index_text.splitlines():
        line = line.strip()
        if not line or len(line) < 3:
            continue
        if line[0].isdigit():
            continue
        name_part = _strip_page_refs(line)
        if len(name_part) < 2:
            continue
        parts = _split_surname_first(name_part)
        if parts:
            surname, rest = parts
            name = f"{rest.strip()} {surname.strip()}"
            type_hint = "Person"
        elif _is_likely_org(name_part):
            name, type_hint = name_part, "Organization"
        elif _is_likely_person(name_part):
            name, type_hint = name_part, "Person"
        else:
            name, type_hint = name_part, None
        name = name.strip()
        if name:
            seeds.append((name, type_hint))
    return seeds


# ── Private helpers ───────────────────────────────────────────────────────────

def _find_isbn(text: str) -> Optional[str]:
    for line in text.splitlines():
        if "ISBN" in line:
            clean = re.sub(r"[^0-9X]", "", line)
            if len(clean) in (10, 13):
                return clean
    return None


def _find_publisher(text: str) -> Optional[str]:
    for line in text.splitlines():
        lower = line.lower()
        if "published by" in lower:
            pos = lower.find("published by")
            after = line[pos + 12:].strip()
            name = re.split(r"[,.]", after)[0].strip()
            if name and len(name) < 80:
                return name
    return None


def _find_copyright_year(text: str) -> Optional[str]:
    for line in text.splitlines():
        if "copyright" in line.lower() or "©" in line:
            buf = ""
            for ch in line:
                if ch.isdigit():
                    buf += ch
                    if len(buf) == 4:
                        y = int(buf)
                        if 1800 <= y <= 2100:
                            return buf
                        buf = ""
                else:
                    buf = ""
    return None


def _find_copyright_holder(text: str) -> Optional[str]:
    for line in text.splitlines():
        if "©" in line:
            pos = line.find("©")
            after = line[pos + 1:].strip()
            after = after.lstrip("opyright").strip()
            holder = ""
            for ch in after:
                if ch.isdigit():
                    break
                holder += ch
            holder = holder.strip().rstrip(",")
            if len(holder) > 2:
                return holder
    return None


def _find_by_author(text: str) -> Optional[str]:
    for line in text.splitlines():
        t = line.strip()
        if t.startswith("By "):
            name = t[3:].strip()
            words = name.split()
            if 2 <= len(words) <= 5 and words[0][0].isupper():
                return name
    return None


def _strip_page_refs(line: str) -> str:
    """Strip trailing page references from an index entry line."""
    for i, c in enumerate(line):
        if c == ",":
            rest = line[i + 1:].strip()
            if rest and all(
                ch.isdigit()
                or ch in "ivxlcIVXLC, -\u2013"
                for ch in rest
            ):
                return line[:i].strip()
    return line.strip()


def _split_surname_first(name: str) -> Optional[tuple[str, str]]:
    """Detect 'Surname, Firstname [Middle]' → (surname, rest)."""
    if "," in name:
        pos = name.index(",")
        surname = name[:pos].strip()
        rest = name[pos + 1:].strip()
        if (
            len(surname.split()) == 1
            and surname[0].isupper()
            and rest and rest[0].isupper()
        ):
            return surname, rest
    return None


def _is_likely_person(name: str) -> bool:
    words = name.split()
    if not (2 <= len(words) <= 4):
        return False
    if not all(w[0].isupper() for w in words if w):
        return False
    place_words = {
        "street", "road", "avenue", "drive", "lane", "place", "square",
        "district", "city", "cape", "town", "province", "mountain", "river",
        "hall", "house", "building", "school", "museum", "station",
    }
    if any(w in place_words for w in name.lower().split()):
        return False
    return True


def _is_likely_org(name: str) -> bool:
    org_keywords = {
        "congress", "committee", "association", "league", "party", "union",
        "council", "institute", "museum", "school", "university", "college",
        "company", "corporation", "limited", "ltd", "inc", "foundation",
        "movement", "government", "ministry", "department", "church", "mosque",
    }
    lower = name.lower()
    if any(k in lower for k in org_keywords):
        return True
    if "(" in name and ")" in name:
        return True
    return False
