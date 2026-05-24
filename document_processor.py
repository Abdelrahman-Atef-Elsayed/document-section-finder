from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import List

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

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


SUPPORTED_EXTENSIONS = ("pdf", "docx", "txt")

SEARCH_TEXT_MAX_CHARS = 1000
PAGE_MARKER_RE = re.compile(r"\[PAGE (\d+)\]")


@dataclass
class Section:
    number: str
    title: str
    content: str
    source: str
    line_number: int
    page_number: int = 0

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "line_number": self.line_number,
            "page_number": self.page_number,
        }


def extract_text_from_pdf(file) -> str:
    if not PDF_AVAILABLE:
        raise RuntimeError("PyPDF2 is not installed")

    reader = PyPDF2.PdfReader(file)
    parts = []
    for page_num, page in enumerate(reader.pages, 1):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            parts.append(f"\n[PAGE {page_num}]\n{page_text}")

    combined = "\n".join(parts)

    if not combined.strip() and OCR_AVAILABLE:
        file.seek(0)
        combined = extract_text_from_pdf_ocr(file)

    return combined


def extract_text_from_pdf_ocr(file) -> str:
    if not OCR_AVAILABLE:
        raise RuntimeError("pytesseract not installed")

    reader = PyPDF2.PdfReader(file)
    parts = []
    for page_num, page in enumerate(reader.pages, 1):
        for img in page.images:
            try:
                image = Image.open(io.BytesIO(img.data))
                text = pytesseract.image_to_string(image)
                if text.strip():
                    parts.append(f"\n[PAGE {page_num}]\n{text}")
            except Exception:
                continue
    return "\n".join(parts)


def extract_text_from_docx(file) -> str:
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx is not installed")
    doc = Document(file)
    parts = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        parts.append(text)
    return "\n\n".join(parts)


def extract_text_from_txt(file) -> str:
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")
    return content


def parse_document(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    if name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    if name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    raise ValueError(f"Unsupported file format: {uploaded_file.name}")


def split_paragraphs(text: str, filename: str = "document") -> List[Section]:
    sections: List[Section] = []
    page_number = 0
    raw_paragraphs = re.split(r"\n\s*\n", text)

    for i, raw in enumerate(raw_paragraphs):
        raw = raw.strip()
        if not raw:
            continue

        lines = raw.split("\n")
        filtered_lines = []
        for line in lines:
            page_marker_match = PAGE_MARKER_RE.match(line.strip())
            if page_marker_match:
                page_number = int(page_marker_match.group(1))
            else:
                filtered_lines.append(line)

        para_text = " ".join(filtered_lines).strip()
        if not para_text:
            continue

        title = para_text[:60] + ("..." if len(para_text) > 60 else "")
        sections.append(Section(
            number=str(i + 1),
            title=title,
            content=para_text,
            source=filename,
            line_number=i + 1,
            page_number=page_number,
        ))

    return sections


def build_sections_dataframe(sections: List[Section]) -> pd.DataFrame:
    if not sections:
        return pd.DataFrame()
    df = pd.DataFrame([s.to_dict() for s in sections])
    df["search_text"] = df["content"].astype(str).str.slice(0, SEARCH_TEXT_MAX_CHARS)
    return df


def missing_optional_libraries() -> list[str]:
    missing = []
    if not PDF_AVAILABLE:
        missing.append("PyPDF2 (PDF support)")
    if not DOCX_AVAILABLE:
        missing.append("python-docx (DOCX support)")
    if not OCR_AVAILABLE:
        missing.append("pytesseract + Pillow (OCR for scanned PDFs)")
    return missing
