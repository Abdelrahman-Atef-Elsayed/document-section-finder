# Document Section Finder

Upload PDF, DOCX, or TXT files, ask a natural-language question, and get back the most relevant paragraphs with their page numbers. Entirely local — no API keys required.

Built with Streamlit, sentence-transformers, ChromaDB, and a free local LLM.

---

## Features

### Core search
- **Multi-format ingestion** — PDF, DOCX, and TXT in a single upload step.
- **Paragraph-level search** — documents are split into paragraphs, each embedded for semantic search.
- **Page number tracking** — PDF page numbers are detected and displayed in results.
- **Hybrid search (BM25 + semantic)** — combines keyword matching with dense embeddings for better recall. Weight adjustable.
- **Cross-encoder re-ranking** — re-ranks top results with `cross-encoder/ms-marco-MiniLM-L-6-v2` for higher precision.
- **Multi-model embedding selector** — choose from 5 free sentence-transformer models (small/fast to high-quality).
- **Tunable relevance** — adjust results count and minimum similarity threshold.
- **Highlighted match snippets** — query-relevant sentences extracted with sentence-level embeddings; query terms highlighted in `<mark>` tags.
- **One-click copy** — copy matched paragraph references to clipboard.

### Advanced
- **RAG (Retrieval-Augmented Generation)** — toggle to generate a natural-language answer from a free local LLM (`google/flan-t5-large`) with source citations.
- **Persistent vector index** — embeddings saved to disk via ChromaDB, surviving app restarts. Save / load with one click.
- **OCR for scanned PDFs** — automatically falls back to `pytesseract` when PyPDF2 extracts no text.
- **PDF report export** — download search results as a formatted PDF document.
- **Keyboard shortcuts** — Ctrl+Enter to search, Esc to clear input.

### Privacy
- **100% local** — no documents leave your machine; no API keys required.
- **No telemetry** — ChromaDB telemetry is disabled.

---

## Architecture

```
app.py                  Streamlit UI (tabs, sidebar, keyboard shortcuts)
├── document_processor  File parsing, OCR, paragraph splitting, page tracking
├── semantic_search     BM25 + semantic hybrid search, cross-encoder re-ranker
├── vector_index        ChromaDB persistent vector storage
├── rag_engine          Free local LLM for AI answer generation
├── export_utils        PDF report generation
└── styles.css          UI styling (light theme)
```

### Workflow

1. **Upload** — drop PDF/DOCX/TXT files into the UI.
2. **Parse** — text extracted per file type; OCR fallback for scanned PDFs.
3. **Split** — text split into paragraphs; PDF page markers tracked.
4. **Embed** — each paragraph encoded into a vector using the selected embedding model.
5. **Search** — query embedded and ranked via hybrid BM25 + cosine similarity; optionally re-ranked by a cross-encoder.
6. **Snippet extraction** — for each result, the most query-relevant sentences are extracted and highlighted.
7. **RAG (optional)** — the top paragraphs are fed to a local LLM for a generated answer with citations.
8. **Render** — results shown as expandable cards with paragraph number, page, source, and highlighted snippet.

---

## Installation

> Requires Python 3.9+.

```bash
git clone https://github.com/Abdelrahman-Atef-Elsayed/document-section-finder.git
cd document-section-finder

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Optional dependencies

| Feature                 | Package                            | Download size                     |
|-------------------------|------------------------------------|-----------------------------------|
| Hybrid search (BM25)    | `rank-bm25`                        | small                             |
| Cross-encoder re-ranker | `sentence-transformers` (built-in) | ~80 MB                            |
| RAG (AI answers)        | `transformers`                     | ~800 MB (flan-t5-large)           |
| Persistent index        | `chromadb`                         | small                             |
| OCR for scanned PDFs    | `pytesseract` + `Pillow`           | small (+ Tesseract system binary) |
| PDF export              | `fpdf2`                            | small                             |

All models download automatically on first use and cache locally.

---

## Usage

```bash
streamlit run app.py
```

Streamlit opens `http://localhost:8501` in your browser.

1. **Upload** tab — drop files and click **Process Files**.
2. **Search** tab — type a query and click **Search** (or Ctrl+Enter).
3. Browse results with paragraph numbers, page numbers, and highlighted snippets.
4. Toggle **Hybrid search**, **Cross-encoder**, or **AI answer** in the sidebar.
5. Click **Export Results as PDF** to download a report.

### Sidebar settings

| Setting         | Description                                |
|-----------------|--------------------------------------------|
| Embedding model | Choose from 5 sentence-transformers models |
| Hybrid search   | BM25 keyword + semantic cosine fusion      |
| Cross-encoder   | Precision re-ranker on top results         |
| AI answer (RAG) | Local LLM generates answer with citations  |

### Persistent index

After processing documents, click **Save to Index** to persist embeddings to disk.
On subsequent launches, click **Load Saved Index** to restore without re-uploading.

### Example queries

- *What are the requirements for user authentication?*
- *Data retention and privacy policies*
- *System architecture and design overview*

---

## Tech Stack

| Layer            | Technology |
|------------------|---------------------------------------------------------------------------------------------------------------------------|
| UI | Streamlit   |
| Embedding models | `paraphrase-MiniLM-L6-v2`, `all-MiniLM-L6-v2`, `multi-qa-MiniLM-L6-dot-v1`, `all-mpnet-base-v2`, `BAAI/bge-small-en-v1.5` |
| Cross-encoder    | `cross-encoder/ms-marco-MiniLM-L-6-v2`                                                                                    |
| RAG LLM          | `google/flan-t5-large` (via Hugging Face Transformers)                                                                    |
| Vector index     | ChromaDB (persistent, local)                                                                                              |
| Tensor backend   | PyTorch (CPU)                                                                                                             |
| Keyword search   | BM25 (rank-bm25)                                                                                                          |
| Data handling    | pandas                                                                                                                    |
| PDF parsing      | PyPDF2 + pytesseract (OCR fallback)                                                                                       |
| DOCX parsing     | python-docx                                                                                                               |
| PDF export       | fpdf2                                                                                                                     |
| Language         | Python 3.9+                                                                                                               |

---

## Project Structure

```
.
├── app.py                  # Streamlit UI entry point
├── document_processor.py   # File parsing, OCR, paragraph splitting
├── semantic_search.py      # Hybrid search, cross-encoder, multi-model
├── vector_index.py         # ChromaDB persistent vector storage
├── rag_engine.py           # Local LLM for AI answer generation
├── export_utils.py         # PDF report export
├── styles.css              # UI styling
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT
└── README.md
```

---

## License

Released under the [MIT License](LICENSE).
