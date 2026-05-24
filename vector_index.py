from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


INDEX_DIR = Path(__file__).parent / ".vector_index"
COLLECTION_NAME = "paragraphs"


def get_client() -> Optional[chromadb.Client]:
    if not CHROMA_AVAILABLE:
        return None
    INDEX_DIR.mkdir(exist_ok=True)
    return chromadb.Client(Settings(
        persist_directory=str(INDEX_DIR),
        anonymized_telemetry=False,
    ))


def index_exists() -> bool:
    if not CHROMA_AVAILABLE:
        return False
    client = get_client()
    if client is None:
        return False
    try:
        client.get_collection(COLLECTION_NAME)
        return True
    except ValueError:
        return False


def get_document_count() -> int:
    if not CHROMA_AVAILABLE or not index_exists():
        return 0
    try:
        client = get_client()
        col = client.get_collection(COLLECTION_NAME)
        return col.count()
    except (ValueError, AttributeError):
        return 0


def store_embeddings(df: pd.DataFrame, embeddings: List[List[float]]) -> bool:
    if not CHROMA_AVAILABLE:
        return False
    try:
        client = get_client()
        col = client.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        ids = df["number"].astype(str).tolist()
        metadatas = []
        for _, row in df.iterrows():
            metadatas.append({
                "number": str(row.get("number", "")),
                "title": str(row.get("title", ""))[:200],
                "source": str(row.get("source", "")),
                "page_number": str(row.get("page_number", 0)),
            })

        texts = df["search_text"].tolist() if "search_text" in df.columns else df["content"].tolist()

        col.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
        )
        return True
    except Exception:
        return False


def load_index_df() -> Optional[pd.DataFrame]:
    if not CHROMA_AVAILABLE or not index_exists():
        return None
    try:
        client = get_client()
        col = client.get_collection(COLLECTION_NAME)
        data = col.get(include=["metadatas", "documents"])
        if not data or not data["ids"]:
            return None
        rows = []
        for i, doc_id in enumerate(data["ids"]):
            meta = data["metadatas"][i] if data["metadatas"] else {}
            doc = data["documents"][i] if data["documents"] else ""
            rows.append({
                "number": meta.get("number", doc_id),
                "title": meta.get("title", doc[:60]),
                "content": doc,
                "source": meta.get("source", ""),
                "page_number": int(meta.get("page_number", 0)),
                "search_text": doc,
            })
        return pd.DataFrame(rows)
    except Exception:
        return None


def get_collection_embeddings() -> Optional[List[List[float]]]:
    if not CHROMA_AVAILABLE or not index_exists():
        return None
    try:
        client = get_client()
        col = client.get_collection(COLLECTION_NAME)
        data = col.get(include=["embeddings"])
        if data and data["embeddings"]:
            return data["embeddings"]
        return None
    except Exception:
        return None


def clear_index() -> bool:
    if not CHROMA_AVAILABLE:
        return False
    try:
        client = get_client()
        client.delete_collection(COLLECTION_NAME)
        return True
    except Exception:
        return False
