"""
chunker.py — Text chunking: character-level sliding window and paragraph-semantic strategies.

Mirrors chunker.rs.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from doc_schema import DocSchema, SectionType, SectionDef, match_section


class ChunkStrategy(Enum):
    Character = auto()   # sliding-window over Unicode scalars (original behaviour)
    Paragraph = auto()   # paragraph → sentence → character cascade (semantic)


class SurrMode(Enum):
    Truncated = auto()   # ±(chunk_size/4) chars from adjacent chunks
    Full = auto()        # complete adjacent chunks included


@dataclass
class ChunkConfig:
    chunk_size: int = 800
    chunk_overlap: int = 200
    min_chunk_len: int = 20
    strategy: ChunkStrategy = ChunkStrategy.Character
    surr_mode: SurrMode = SurrMode.Truncated


@dataclass
class Chunk:
    id: int
    text: str
    surrounding: str
    doc_name: str
    chunk_index: int
    page_num: Optional[int] = None
    section_name: Optional[str] = None
    skip_extraction: bool = False
    section_note: Optional[str] = None
    section_type: SectionType = SectionType.Main


def chunk_id(doc_name: str, chunk_index: int) -> int:
    """Deterministic stable chunk ID (mirrors Rust sha256 hash)."""
    h = hashlib.sha256()
    h.update(doc_name.encode())
    h.update(b"::")
    h.update(struct.pack("<I", chunk_index))
    digest = h.digest()
    value = struct.unpack_from("<q", digest[:8])[0]
    return value


def split_text(
    text: str,
    doc_name: str,
    cfg: Optional[ChunkConfig] = None,
    schema: Optional[DocSchema] = None,
) -> list[Chunk]:
    if cfg is None:
        cfg = ChunkConfig()
    if cfg.strategy == ChunkStrategy.Character:
        return _split_character(text, doc_name, cfg)
    else:
        return _split_paragraph(text, doc_name, cfg, schema)


# ── Character strategy ────────────────────────────────────────────────────────

def _split_character(text: str, doc_name: str, cfg: ChunkConfig) -> list[Chunk]:
    chars = list(text)
    total = len(chars)
    if total == 0:
        return []

    step = max(cfg.chunk_size - cfg.chunk_overlap, 1)
    chunks = []
    pos = 0
    index = 0

    while pos < total:
        end = min(pos + cfg.chunk_size, total)
        text_str = "".join(chars[pos:end])

        surr_half = cfg.chunk_size // 4
        surr_start = max(0, pos - surr_half)
        surr_end = min(end + surr_half, total)
        surrounding = "".join(chars[surr_start:surr_end])

        if len(text_str) >= cfg.min_chunk_len:
            chunks.append(Chunk(
                id=chunk_id(doc_name, index),
                text=text_str,
                surrounding=surrounding,
                doc_name=doc_name,
                chunk_index=index,
            ))
            index += 1

        pos += step

    return chunks


# ── Paragraph strategy ────────────────────────────────────────────────────────

def _split_paragraph(
    text: str,
    doc_name: str,
    cfg: ChunkConfig,
    schema: Optional[DocSchema],
) -> list[Chunk]:
    units = _collect_units_with_headings(text, cfg)
    if not units:
        return []

    cur_section_name: Optional[str] = None
    cur_skip = False
    cur_note: Optional[str] = None
    cur_section_type = SectionType.Main

    content_units: list[tuple[str, Optional[str], bool, Optional[str], SectionType]] = []

    for is_heading, unit_text in units:
        if is_heading:
            if schema:
                sec = match_section(unit_text, schema)
                if sec:
                    cur_section_name = unit_text
                    cur_skip = sec.skip
                    cur_note = sec.narrator_note
                    cur_section_type = sec.section_type
                else:
                    cur_section_name = unit_text
                    cur_skip = False
                    cur_note = None
                    cur_section_type = SectionType.Main
            else:
                cur_section_name = unit_text
        else:
            content_units.append((unit_text, cur_section_name, cur_skip, cur_note, cur_section_type))

    packed = _pack_chunks_with_meta(content_units, cfg)
    surr_half = cfg.chunk_size // 4
    chunk_texts = [t for t, *_ in packed] #Comprehension 

    result = []
    index = 0

    for i, (text_str, sec_name, skip, note, sec_type) in enumerate(packed):
        if len(text_str) < cfg.min_chunk_len:
            continue

        if cfg.surr_mode == SurrMode.Full:
            parts = []
            if i > 0:
                parts.append(chunk_texts[i - 1])
                parts.append(" ")
            parts.append(text_str)
            if i + 1 < len(chunk_texts):
                parts.append(" ")
                parts.append(chunk_texts[i + 1])
            surrounding = "".join(parts)
        else:  # Truncated
            parts = []
            if i > 0:
                prev = list(chunk_texts[i - 1])
                tail_start = max(0, len(prev) - surr_half)
                parts.append("".join(prev[tail_start:]))
                parts.append(" ")
            parts.append(text_str)
            if i + 1 < len(chunk_texts):
                nxt = list(chunk_texts[i + 1])
                head_end = min(surr_half, len(nxt))
                parts.append(" ")
                parts.append("".join(nxt[:head_end]))
            surrounding = "".join(parts)

        result.append(Chunk(
            id=chunk_id(doc_name, index),
            text=text_str,
            surrounding=surrounding,
            doc_name=doc_name,
            chunk_index=index,
            section_name=sec_name,
            skip_extraction=skip,
            section_note=note,
            section_type=sec_type,
        ))
        index += 1

    return result


def _pack_chunks_with_meta(
    units: list[tuple[str, Optional[str], bool, Optional[str], SectionType]],
    cfg: ChunkConfig,
) -> list[tuple[str, Optional[str], bool, Optional[str], SectionType]]:
    result = []
    parts: list[str] = []
    cur_len = 0
    cur_meta: tuple[Optional[str], bool, Optional[str], SectionType] = (None, False, None, SectionType.Main)

    def emit():
        if parts:
            result.append(("\n".join(parts), *cur_meta))

    for unit_text, sec_name, skip, note, sec_type in units:
        unit_len = len(unit_text)
        sep = 1 if parts else 0
        same_zone = not parts or cur_meta[3].same_window_zone(sec_type)

        if same_zone and (not parts or cur_len + sep + unit_len <= cfg.chunk_size):
            if not parts:
                cur_meta = (sec_name, skip, note, sec_type)
            cur_len += sep + unit_len
            parts.append(unit_text)
        else:
            emit()
            prev_zone_same = bool(result) and result[-1][4].same_window_zone(sec_type)
            if prev_zone_same:
                prev_chars = list(result[-1][0])
                ol_start = max(0, len(prev_chars) - cfg.chunk_overlap)
                overlap = "".join(prev_chars[ol_start:])
            else:
                overlap = ""

            parts.clear()
            cur_len = 0
            cur_meta = (sec_name, skip, note, sec_type)

            if overlap:
                cur_len += len(overlap) + 1
                parts.append(overlap)
            cur_len += unit_len
            parts.append(unit_text)

    emit()
    return result


def _collect_units_with_headings(text: str, cfg: ChunkConfig) -> list[tuple[bool, str]]:
    """Return (is_heading, text) pairs for all paragraphs."""
    result: list[tuple[bool, str]] = []
    acc = ""

    def flush():
        nonlocal acc
        if not acc:
            return
        if len(acc) >= cfg.min_chunk_len:
            result.append((False, acc))
        elif result:
            # Merge into previous content unit
            is_h, prev = result[-1]
            result[-1] = (is_h, prev + "\n" + acc)
        acc = ""

    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue

        trimmed = para.rstrip()
        is_heading = (
            para.count("\n") == 0
            and len(para) <= 80
            and not trimmed.endswith(".")
            and not trimmed.endswith("!")
            and not trimmed.endswith("?")
        )

        if is_heading:
            flush()
            result.append((True, para))
        elif len(para) <= cfg.chunk_size:
            if len(para) < cfg.min_chunk_len:
                acc = (acc + "\n" + para).lstrip("\n") if acc else para
            else:
                flush()
                result.append((False, para))
        else:
            flush()
            out: list[str] = []
            _split_sentences(para, cfg, out)
            for s in out:
                result.append((False, s))

    flush()
    return result


def _split_sentences(text: str, cfg: ChunkConfig, out: list[str]) -> None:
    chars = list(text)
    n = len(chars)
    start = 0
    i = 0

    while i < n:
        if (
            chars[i] in ".!?"
            and i + 2 < n
            and chars[i + 1].isspace()
            and chars[i + 2].isalpha()
        ):
            seg = "".join(chars[start: i + 1]).strip()
            if len(seg) > cfg.chunk_size:
                _split_chars(seg, cfg, out)
            elif len(seg) >= cfg.min_chunk_len:
                out.append(seg)
            start = i + 2
            i = start
            continue
        i += 1

    if start < n:
        tail = "".join(chars[start:]).strip()
        if len(tail) > cfg.chunk_size:
            _split_chars(tail, cfg, out)
        elif len(tail) >= cfg.min_chunk_len:
            out.append(tail)


def _split_chars(text: str, cfg: ChunkConfig, out: list[str]) -> None:
    chars = list(text)
    step = max(cfg.chunk_size - cfg.chunk_overlap, 1)
    pos = 0
    while pos < len(chars):
        end = min(pos + cfg.chunk_size, len(chars))
        s = "".join(chars[pos:end])
        if len(s) >= cfg.min_chunk_len:
            out.append(s)
        pos += step
