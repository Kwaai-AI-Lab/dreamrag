"""
document.py — Extract plain text from document files.

Supported formats:
  - .txt, .md, .rst, .csv, .json, .yaml, .toml  — read as UTF-8
  - .pdf  — extracted via pdfminer.six (pip install pdfminer.six)
  - .docx — extracted from embedded XML via zipfile (no extra deps)
  - .doc  — extracted via antiword or libreoffice --headless (must be installed)
"""

_OOXML_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

from __future__ import annotations

import re
import subprocess
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

SUPPORTED_EXTENSIONS = [
    "txt", "md", "rst", "csv", "json", "yaml", "toml", "pdf", "docx", "doc",
]


def extract_text(path: str | Path) -> str:
    """Extract plain text from a document file."""
    path = Path(path)
    ext = path.suffix.lstrip(".").lower()

    if ext == "pdf":
        return _extract_pdf(path)
    elif ext == "docx":
        return _extract_docx(path)
    elif ext == "doc":
        return _extract_doc_legacy(path)
    else:
        return path.read_text(encoding="utf-8")


# ── PDF ───────────────────────────────────────────────────────────────────────

def _extract_pdf(path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
    except ImportError:
        raise RuntimeError(
            "PDF support requires pdfminer.six:\n  pip install pdfminer.six"
        )
    try:
        text = pdfminer_extract(str(path))
        return _clean_pdf_text(text or "")
    except Exception as e:
        raise RuntimeError(f"extracting PDF text from {path}: {e}") from e


def _clean_pdf_text(text: str) -> str:
    """Fix underscore artifacts introduced by some PDF glyph-to-Unicode mappings.

    Rules (applied per `_` character):
      1. `_s` at a word boundary → `'s`
      2. `_` preceded by a letter and followed by whitespace, end, or uppercase → `.`
      3. All other underscores → stripped
    """
    chars = list(text)
    n = len(chars)
    out = []
    i = 0
    while i < n:
        c = chars[i]
        if c != "_":
            out.append(c)
            i += 1
            continue

        # Rule 1: `_s` where s is followed by non-alpha or end → apostrophe-s
        if i + 1 < n and chars[i + 1] == "s":
            after = i + 2
            at_boundary = after >= n or not chars[after].isalpha()
            if at_boundary:
                out.append("'")
                out.append("s")
                i += 2
                continue

        # Rule 2: `_` preceded by a letter and followed by whitespace, end, or uppercase → `.`
        prev_is_alpha = bool(out) and out[-1].isalpha()
        next_char = chars[i + 1] if i + 1 < n else None
        next_is_break_or_initial = (
            next_char is None
            or next_char in (" ", "\n", "\r")
            or (next_char.isalpha() and next_char.isupper())
        )
        if prev_is_alpha and next_is_break_or_initial:
            out.append(".")
            i += 1
            continue

        # Rule 3: strip
        i += 1

    return "".join(out)


# ── DOCX ──────────────────────────────────────────────────────────────────────




def _extract_docx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            with zf.open("word/document.xml") as f:
                xml_bytes = f.read()
    except KeyError:
        raise RuntimeError("word/document.xml not found — is this a valid .docx file?")
    except Exception as e:
        raise RuntimeError(f"reading DOCX archive {path}: {e}") from e

    return _parse_ooxml_text(xml_bytes.decode("utf-8"))


def _parse_ooxml_text(xml: str) -> str:
    """Extract text from Office Open XML (word/document.xml).

    Reads <w:t> elements; inserts newlines at paragraph (<w:p>) and run breaks.
    """
    ns = _OOXML_NS
    root = ET.fromstring(xml)

    out = []

    def _walk(elem: ET.Element) -> None:
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

        if tag == "p":
            if out and not out[-1].endswith("\n"):
                out.append("\n")
        elif tag in ("br", "cr"):
            out.append("\n")
        elif tag == "tab":
            out.append("\t")
        elif tag == "t":
            out.append(elem.text or "")

        for child in elem:
            _walk(child)

    _walk(root)
    return "".join(out)


# ── Legacy .doc ───────────────────────────────────────────────────────────────

def _extract_doc_legacy(path: Path) -> str:
    # Try antiword first (lighter, faster).
    try:
        result = subprocess.run(
            ["antiword", str(path)],
            capture_output=True,
            check=True,
        )
        return result.stdout.decode("utf-8", errors="replace")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fall back to LibreOffice headless conversion.
    tmp = Path(tempfile.mkdtemp(prefix="kwaai-doc-convert-"))
    try:
        result = subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to", "txt:Text",
                "--outdir", str(tmp),
                str(path),
            ],
            capture_output=True,
        )
        if result.returncode == 0:
            txt = (tmp / path.stem).with_suffix(".txt")
            if txt.exists():
                return txt.read_text(encoding="utf-8")
    except FileNotFoundError:
        pass

    raise RuntimeError(
        f"Cannot extract text from '{path}'. "
        "Install antiword (brew install antiword) or LibreOffice."
    )
