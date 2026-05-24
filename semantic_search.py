from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


AVAILABLE_MODELS = {
    "paraphrase-MiniLM-L6-v2": "Fast, general purpose",
    "all-MiniLM-L6-v2": "General purpose (good quality)",
    "multi-qa-MiniLM-L6-dot-v1": "Best for Q&A",
    "all-mpnet-base-v2": "Best quality (larger)",
    "BAAI/bge-small-en-v1.5": "Very fast, good quality",
}

DEFAULT_MODEL_NAME = "paraphrase-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CONTENT_PREVIEW_CHARS = 500
DEFAULT_SNIPPET_CHARS = 500

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
    score_bm25: float = 0.0
    score_semantic: float = 0.0
    score_rerank: float = 0.0
    number: str = ""
    title: str = ""
    content: str = ""
    source: str = ""
    page_number: int = 0
    snippet_html: str = ""

    def to_dict(self) -> dict:
        return {
            "similarity": self.similarity,
            "number": self.number,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "page_number": self.page_number,
            "snippet_html": self.snippet_html,
        }


_cross_encoder_model = None


def load_model(model_name: str = DEFAULT_MODEL_NAME, device: str = "cpu") -> SentenceTransformer:
    model = SentenceTransformer(model_name, device=device)
    with torch.no_grad():
        _ = model.encode("warmup", convert_to_tensor=True)
    return model


def load_cross_encoder(device: str = "cpu") -> Optional:
    global _cross_encoder_model
    if not CROSS_ENCODER_AVAILABLE:
        return None
    if _cross_encoder_model is None:
        _cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
    return _cross_encoder_model


def encode_corpus(model: SentenceTransformer, df: pd.DataFrame) -> Optional[torch.Tensor]:
    if df.empty:
        return None
    texts = [t if isinstance(t, str) and t.strip() else "N/A" for t in df["search_text"].tolist()]
    return model.encode(texts, convert_to_tensor=True, show_progress_bar=False)


def _prepare_bm25(df: pd.DataFrame) -> Optional:
    if not BM25_AVAILABLE:
        return None
    texts = df["search_text"].tolist() if "search_text" in df.columns else df["content"].tolist()
    tokenized = [_tokenize(t) for t in texts]
    return BM25Okapi(tokenized)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def search(
    model: SentenceTransformer,
    query: str,
    df: pd.DataFrame,
    embeddings: Optional[torch.Tensor],
    top_k: int = 10,
    min_similarity: float = 0.3,
    snippet_chars: int = DEFAULT_SNIPPET_CHARS,
    use_hybrid: bool = True,
    use_cross_encoder: bool = False,
    bm25_weight: float = 0.3,
) -> Tuple[List[SearchResult], List[str]]:
    if df.empty or not query.strip():
        return [], []

    num_docs = len(df)
    effective_k = min(top_k * 3, num_docs)

    bm25_scores = None
    if use_hybrid and BM25_AVAILABLE:
        bm25 = _prepare_bm25(df)
        if bm25:
            tokenized_query = _tokenize(query)
            raw_scores = bm25.get_scores(tokenized_query)
            raw_scores = np.where(raw_scores < 0, 0, raw_scores)
            max_score = raw_scores.max() if raw_scores.max() > 0 else 1.0
            bm25_scores = raw_scores / max_score

    semantic_scores = None
    if embeddings is not None:
        query_emb = model.encode(query, convert_to_tensor=True)
        semantic_scores = util.pytorch_cos_sim(query_emb, embeddings)[0].cpu().numpy()

    combined_scores = None
    if semantic_scores is not None and bm25_scores is not None:
        combined_scores = (1 - bm25_weight) * semantic_scores + bm25_weight * bm25_scores
    elif semantic_scores is not None:
        combined_scores = semantic_scores
    elif bm25_scores is not None:
        combined_scores = bm25_scores
    else:
        return [], []

    top_indices = np.argsort(combined_scores)[::-1][:effective_k]

    df_subset = df.iloc[top_indices].copy()

    if use_cross_encoder and CROSS_ENCODER_AVAILABLE:
        ce_model = load_cross_encoder()
        if ce_model:
            pairs = [[query, str(df_subset.iloc[i]["search_text"])] for i in range(len(df_subset))]
            ce_scores = ce_model.predict(pairs)
            for i in range(len(df_subset)):
                df_subset.iloc[i, df_subset.columns.get_loc("_ce_score") if "_ce_score" in df_subset.columns else -1] = 0
            df_subset["_ce_score"] = ce_scores
            df_subset = df_subset.sort_values("_ce_score", ascending=False)
            top_indices_final = df_subset.index[:top_k]
        else:
            hybrid_order = np.argsort(combined_scores[top_indices])[::-1]
            top_indices_final = top_indices[hybrid_order][:top_k]
    else:
        hybrid_order = np.argsort(combined_scores[top_indices])[::-1]
        top_indices_final = top_indices[hybrid_order][:top_k]

    results: List[SearchResult] = []
    numbers: List[str] = []

    for idx in top_indices_final:
        row = df.iloc[idx]
        final_score = combined_scores[idx]
        sem_score = float(semantic_scores[idx]) if semantic_scores is not None else 0.0
        bm25_val = float(bm25_scores[idx]) if bm25_scores is not None else 0.0

        if final_score < min_similarity:
            continue

        full_content = str(row.get("content", "N/A"))
        ce_score = 0.0
        if use_cross_encoder and CROSS_ENCODER_AVAILABLE and "_ce_score" in df_subset.columns:
            lc = df_subset.loc[df_subset.index == idx]
            if not lc.empty:
                ce_score = float(lc["_ce_score"].iloc[0])

        query_emb = None
        if embeddings is not None:
            query_emb = model.encode(query, convert_to_tensor=True)
            snippet = (
                extract_snippet(model, query_emb, full_content, max_chars=snippet_chars)
                if full_content and full_content != "N/A"
                else ""
            )
        else:
            snippet = ""

        snippet_html = highlight_terms(snippet, query) if snippet else ""

        result = SearchResult(
            similarity=round(float(final_score), 3),
            score_bm25=round(bm25_val, 3),
            score_semantic=round(sem_score, 3),
            score_rerank=round(float(ce_score), 3),
            number=str(row.get("number", "N/A")),
            title=str(row.get("title", "N/A")),
            content=full_content[:CONTENT_PREVIEW_CHARS],
            source=str(row.get("source", "N/A")),
            page_number=int(row.get("page_number", 0)),
            snippet_html=snippet_html,
        )
        results.append(result)
        numbers.append(result.number)

    return results[:top_k], numbers[:top_k]


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

def _split_sentences(text: str) -> List[str]:
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


_WORD_RE = re.compile(r"\b\w+\b")


def highlight_terms(text: str, query: str) -> str:
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
    if not section_numbers:
        return "No matching paragraphs found"
    seen = set()
    unique = []
    for num in section_numbers:
        if num not in seen:
            seen.add(num)
            unique.append(num)
    return " , ".join(unique)


def similarity_badge(score: float) -> str:
    if score >= 0.7:
        return f"🟢 **{score:.1%}**"
    if score >= 0.5:
        return f"🟡 **{score:.1%}**"
    return f"🟠 **{score:.1%}**"
