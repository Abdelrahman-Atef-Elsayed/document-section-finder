"""Document parsing and section extraction.

Handles PDF, DOCX, and TXT files. Detects numbered sections (e.g. 1.2, 3.7.1,
Chapter 4, Article 2) and falls back to heuristic detection when a document
has no explicit numbering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pandas as pd

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


SUPPORTED_EXTENSIONS = ("pdf", "docx", "txt")

SEARCH_TEXT_MAX_CHARS = 1000

SECTION_PATTERNS = [
    r"^[\s]*(\d+(?:\.\d+)+)[\s]*[:\.\-]?[\s]*(.+?)$",
    r"^[\s]*(0(?:\.\d+)+)[\s]*[:\.\-]?[\s]*(.+?)$",
    r"^[\s]*(\d+)\.[\s]+(.+?)$",
    r"^[\s]*(?:Section|SECTION|sec\.?|SEC\.?)[\s]*(\d+(?:\.\d+)*)[\s]*[:\.\-]?[\s]*(.+?)$",
    r"^[\s]*(?:Chapter|CHAPTER|Part|PART)[\s]*(\d+(?:\.\d+)*)[\s]*[:\.\-]?[\s]*(.+?)$",
    r"^[\s]*(?:Article|ARTICLE|Art\.?)[\s]*(\d+(?:\.\d+)*)[\s]*[:\.\-]?[\s]*(.+?)$",
]

HEADING_MARKER_PATTERN = re.compile(r"\[HEADING\](.+?)\[/HEADING\]")


@dataclass
class Section:
    number: str
    title: str
    content: str
    source: str
    line_number: int

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "line_number": self.line_number,
        }


# Text extraction

def extract_text_from_pdf(file) -> str:
    if not PDF_AVAILABLE:
        raise RuntimeError("PyPDF2 is not installed. Install with: pip install PyPDF2")
    reader = PyPDF2.PdfReader(file)
    parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            parts.append(page_text)
    return "\n".join(parts)


def extract_text_from_docx(file) -> str:
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx is not installed. Install with: pip install python-docx")
    doc = Document(file)
    parts = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style.name.startswith("Heading"):
            parts.append(f"\n[HEADING]{text}[/HEADING]\n")
        else:
            parts.append(text)
    return "\n".join(parts)


def extract_text_from_txt(file) -> str:
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")
    return content


def parse_document(uploaded_file) -> str:
    """Dispatch to the right extractor based on file extension."""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    if name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    if name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    raise ValueError(f"Unsupported file format: {uploaded_file.name}")


# Section detection

def _save_current(sections: List[Section], current: Optional[Section], buffer: List[str]) -> None:
    if current is None:
        return
    current.content = " ".join(buffer).strip()
    if current.content or current.title:
        sections.append(current)


def detect_sections(text: str, filename: str = "document") -> List[Section]:
    """Detect numbered sections and headings in text."""
    sections: List[Section] = []
    current: Optional[Section] = None
    buffer: List[str] = []

    lines = text.split("\n")
    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        heading_match = HEADING_MARKER_PATTERN.search(line)
        if heading_match:
            _save_current(sections, current, buffer)
            heading_text = heading_match.group(1).strip()
            num_match = re.match(r"^(\d+(?:\.\d+)*)", heading_text)
            if num_match:
                number = num_match.group(1)
                title = heading_text[len(number):].strip(" :.-")
            else:
                number = f"H{len(sections) + 1}"
                title = heading_text
            current = Section(number=number, title=title, content="", source=filename, line_number=i + 1)
            buffer = []
            continue

        matched = False
        for pattern in SECTION_PATTERNS:
            m = re.match(pattern, line, re.IGNORECASE)
            if not m:
                continue
            _save_current(sections, current, buffer)
            number = m.group(1)
            title = m.group(2).strip() if len(m.groups()) > 1 else ""
            current = Section(number=number, title=title, content="", source=filename, line_number=i + 1)
            buffer = []
            matched = True
            break

        if matched:
            continue

        if current is None:
            current = Section(
                number="0",
                title="Introduction / Preamble",
                content="",
                source=filename,
                line_number=1,
            )
        buffer.append(line)

    _save_current(sections, current, buffer)

    if not sections:
        sections = _detect_sections_heuristic(text, filename)
    return sections


def _detect_sections_heuristic(text: str, filename: str) -> List[Section]:
    """Fallback for documents without numbered sections."""
    sections: List[Section] = []
    paragraphs = re.split(r"\n\s*\n|\r\n\s*\r\n", text)

    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue

        lines = para.split("\n")
        first = lines[0].strip()
        looks_like_heading = (
            len(first) < 100
            and (first.isupper() or first.istitle() or re.match(r"^[\d\.]+", first) or first.endswith(":"))
        )

        if looks_like_heading and len(lines) > 1:
            num_match = re.match(r"^([\d\.]+)", first)
            number = num_match.group(1) if num_match else f"{i + 1}"
            sections.append(
                Section(
                    number=number,
                    title=first,
                    content="\n".join(lines[1:]).strip(),
                    source=filename,
                    line_number=i + 1,
                )
            )
        else:
            sections.append(
                Section(
                    number=f"P{i + 1}",
                    title=first[:50] + ("..." if len(first) > 50 else ""),
                    content=para,
                    source=filename,
                    line_number=i + 1,
                )
            )
    return sections


def build_sections_dataframe(sections: List[Section]) -> pd.DataFrame:
    """Convert sections into a DataFrame with a precomputed search_text column."""
    if not sections:
        return pd.DataFrame()
    df = pd.DataFrame([s.to_dict() for s in sections])
    df["search_text"] = (df["title"].astype(str) + " " + df["content"].astype(str)).str.slice(0, SEARCH_TEXT_MAX_CHARS)
    return df


def missing_optional_libraries() -> list[str]:
    """Return a human-readable list of optional parsers that aren't installed."""
    missing = []
    if not PDF_AVAILABLE:
        missing.append("PyPDF2 (PDF support)")
    if not DOCX_AVAILABLE:
        missing.append("python-docx (DOCX support)")
    return missing
