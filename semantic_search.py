"""Semantic search: embedding model loading and similarity search."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

DEFAULT_MODEL_NAME = "paraphrase-MiniLM-L6-v2"
DEFAULT_DEVICE = "cpu"
CONTENT_PREVIEW_CHARS = 500
DEFAULT_SNIPPET_CHARS = 500

# Common English stopwords filtered out before highlighting.
_HIGHLIGHT_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "for", "of", "to", "in", "on", "at", "by", "with", "as", "from", "into",
        "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
        "what", "how", "why", "when", "where", "which", "who", "whom",
        "do", "does", "did", "can", "could", "should", "would", "will", "shall",
        "have", "has", "had", "i", "you", "he", "she", "it", "we", "they",
        "my", "your", "his", "her", "its", "our", "their",
        "not", "no", "yes", "any", "all", "some",
    }
)


@dataclass
class SearchResult:
    similarity: float
    number: str
    title: str
    content: str
    source: str
    snippet_html: str = ""

    def to_dict(self) -> dict:
        return {
            "similarity": self.similarity,
            "number": self.number,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "snippet_html": self.snippet_html,
        }


def load_model(model_name: str = DEFAULT_MODEL_NAME, device: str = DEFAULT_DEVICE) -> SentenceTransformer:
    """Load the sentence-transformer model and warm it up."""
    model = SentenceTransformer(model_name, device=device)
    with torch.no_grad():
        _ = model.encode("warmup", convert_to_tensor=True)
    return model


def encode_corpus(model: SentenceTransformer, df: pd.DataFrame) -> Optional[torch.Tensor]:
    """Encode the search_text column of a DataFrame into a tensor of embeddings."""
    if df.empty:
        return None
    texts = [t if isinstance(t, str) and t.strip() else "N/A" for t in df["search_text"].tolist()]
    return model.encode(texts, convert_to_tensor=True, show_progress_bar=False)


def search(
    model: SentenceTransformer,
    query: str,
    df: pd.DataFrame,
    embeddings: Optional[torch.Tensor],
    top_k: int = 10,
    min_similarity: float = 0.3,
    snippet_chars: int = DEFAULT_SNIPPET_CHARS,
) -> Tuple[List[SearchResult], List[str]]:
    """Return ranked SearchResult list plus the matching section numbers.

    Each result includes a `snippet_html` field: the most query-relevant
    sentences from the section content with matched query terms wrapped in
    ``<mark>`` tags. Safe to render with ``unsafe_allow_html=True``.
    """
    if df.empty or embeddings is None or not query.strip():
        return [], []

    query_emb = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
    top = torch.topk(scores, k=min(top_k, len(scores)))

    results: List[SearchResult] = []
    numbers: List[str] = []
    for score, idx in zip(top.values, top.indices):
        value = score.item()
        if value < min_similarity:
            continue
        row = df.iloc[idx.item()]
        full_content = str(row.get("content", "N/A"))
        snippet = (
            extract_snippet(model, query_emb, full_content, max_chars=snippet_chars)
            if full_content and full_content != "N/A"
            else ""
        )
        snippet_html = highlight_terms(snippet, query) if snippet else ""

        result = SearchResult(
            similarity=round(value, 3),
            number=str(row.get("number", "N/A")),
            title=str(row.get("title", "N/A")),
            content=full_content[:CONTENT_PREVIEW_CHARS],
            source=str(row.get("source", "N/A")),
            snippet_html=snippet_html,
        )
        results.append(result)
        numbers.append(result.number)
    return results, numbers


# Snippet extraction and highlighting

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_WORD_RE = re.compile(r"\b\w+\b")


def _split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter — punctuation first, newlines as fallback."""
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    if len(parts) == 1 and "\n" in text:
        parts = text.splitlines()
    return [p.strip() for p in parts if p and p.strip()]


def extract_snippet(
    model: SentenceTransformer,
    query_emb: torch.Tensor,
    content: str,
    max_chars: int = DEFAULT_SNIPPET_CHARS,
) -> str:
    """Pick the most query-relevant span (one or more sentences) from content."""
    if not content:
        return ""
    if len(content) <= max_chars:
        return content

    sentences = _split_sentences(content)
    if len(sentences) <= 1:
        return content[:max_chars].rstrip() + "..."

    sentence_embs = model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
    sim = util.pytorch_cos_sim(query_emb, sentence_embs)[0]
    ranked = sorted(range(len(sentences)), key=lambda i: -sim[i].item())

    selected: set[int] = set()
    total = 0
    for idx in ranked:
        s_len = len(sentences[idx]) + 1
        if total and total + s_len > max_chars:
            continue
        selected.add(idx)
        total += s_len
        if total >= max_chars:
            break

    if not selected:
        best = max(range(len(sentences)), key=lambda i: sim[i].item())
        return sentences[best][:max_chars].rstrip() + "..."

    ordered = sorted(selected)
    pieces: List[str] = []
    for pos, i in enumerate(ordered):
        if pos > 0 and i != ordered[pos - 1] + 1:
            pieces.append("…")
        pieces.append(sentences[i])
    return " ".join(pieces)


def highlight_terms(text: str, query: str) -> str:
    """HTML-escape `text` and wrap query terms in ``<mark>`` tags.

    Stopwords and very short tokens are ignored. Output is safe HTML — the text
    is escaped first, and only the ``<mark>`` tags we add are unescaped.
    """
    escaped = html.escape(text)
    raw_terms = _WORD_RE.findall(query.lower())
    terms = {t for t in raw_terms if len(t) > 2 and t not in _HIGHLIGHT_STOPWORDS}
    if not terms:
        return escaped

    sorted_terms = sorted(terms, key=len, reverse=True)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in sorted_terms) + r")\b",
        re.IGNORECASE,
    )
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped)


def format_section_references(section_numbers: List[str]) -> str:
    """Format unique section numbers as a comma-separated string."""
    if not section_numbers:
        return "No matching sections found"
    seen = set()
    unique = []
    for num in section_numbers:
        if num not in seen:
            seen.add(num)
            unique.append(num)
    return " , ".join(unique)


def similarity_badge(score: float) -> str:
    """Return a markdown badge string colour-coded by score band."""
    if score >= 0.7:
        return f"🟢 **{score:.1%}**"
    if score >= 0.5:
        return f"🟡 **{score:.1%}**"
    return f"🟠 **{score:.1%}**"
