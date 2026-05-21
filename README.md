# Document Section Finder

A semantic search engine for structured documents. Upload PDF, DOCX, or TXT
files, ask a natural-language question, and get back the **exact section
numbers** (e.g. `3.7, 0.1.1, 0.6.5.1`) that best match your query — no need to
read through hundreds of pages.

Built with Streamlit and sentence-transformers; runs entirely on your own
machine.

---

## Features

- **Multi-format ingestion** — PDF, DOCX, and TXT in a single upload step.
- **Smart section detection** — recognises numbered sections (`1.2`, `3.7.1`,
  `0.6.5.1`), `Section X`, `Chapter Y`, `Article Z`, and Word heading styles,
  with a heuristic fallback for unstructured documents.
- **Semantic search** — uses sentence-transformer embeddings so queries match
  on *meaning*, not just keywords.
- **Tunable relevance** — adjust the number of results and minimum similarity
  threshold from the sidebar.
- **Detailed result view** — preview each matching section with its title,
  source document, and similarity score.
- **Highlighted match snippets** — for every result, the most query-relevant
  sentences from the section are extracted with sentence-level embeddings and
  query terms are visually highlighted, so the *why* of the match is obvious
  at a glance.
- **One-click copy** — copy the matched section references to the clipboard.
- **CSV export** — download all extracted sections for downstream use.
- **Local-only processing** — no documents leave your machine; no API keys
  required.

---

## Architecture

```
┌──────────────────────┐
│       app.py         │   Streamlit UI: tabs, sidebar, session state
└──────────┬───────────┘
           │
   ┌───────┴───────────────┐
   ▼                       ▼
document_processor   semantic_search
   .py                  .py
   │                    │
   ├ parse_document     ├ load_model          (sentence-transformers)
   ├ detect_sections    ├ encode_corpus       (torch tensors)
   └ build_dataframe    ├ search              (cosine similarity)
                        ├ extract_snippet     (sentence-level relevance)
                        ├ highlight_terms     (HTML-safe <mark> highlighting)
                        └ format_references
```

### Workflow

1. **Upload** — user drops one or more PDF/DOCX/TXT files into the UI.
2. **Parse** — `document_processor` extracts raw text per file type.
3. **Detect sections** — regex patterns + heading markers split the text into
   `Section` records (number, title, content, source).
4. **Embed** — `semantic_search.encode_corpus` turns each section into a
   vector using `paraphrase-MiniLM-L6-v2`.
5. **Search** — the query is embedded the same way and ranked against the
   corpus with cosine similarity.
6. **Snippet extraction** — for each top result, the section content is split
   into sentences, each sentence is embedded against the query, and the
   highest-scoring sentences (up to ~500 chars) are stitched back together in
   document order.
7. **Highlight & render** — query keywords (minus stopwords) are wrapped in
   `<mark>` tags after HTML-escaping. Matching section numbers are surfaced
   as a compact, copyable reference list, with highlighted snippets shown in
   expandable details.

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

The first run downloads the sentence-transformer model (~80 MB) from
Hugging Face and caches it locally.

---

## Usage

```bash
streamlit run app.py
```

Streamlit will open `http://localhost:8501` in your browser.

1. Go to **Upload Documents**, drop your files, and click **Process Files**.
2. Switch to **Search**, type a natural-language question, and click **Search**.
3. Copy the matched references, or browse the detailed previews below.
4. Use **View Sections** to browse all extracted sections or export to CSV.

### Example queries

- *What are the requirements for user authentication?*
- *Data retention and privacy policies*
- *Compliance procedures for security audits*

---

## Tech Stack

| Layer            | Technology                                        |
|------------------|---------------------------------------------------|
| UI               | Streamlit                                         |
| Embedding model  | `sentence-transformers` (paraphrase-MiniLM-L6-v2) |
| Tensor backend   | PyTorch (CPU)                                     |
| Data handling    | pandas                                            |
| PDF parsing      | PyPDF2                                            |
| DOCX parsing     | python-docx                                       |
| Language         | Python 3.9+                                       |

---

## Security Notes

- **No secrets in the repo.** No API keys, tokens, or credentials are required
  to run this project.
- **Local processing only.** Uploaded documents are held in Streamlit's
  in-memory session and never written to disk by the app itself. Nothing is
  sent to a remote service.
- **Sanitised clipboard payload.** The "Copy" button serialises the reference
  string with `json.dumps` before embedding it in JavaScript, so crafted
  section text cannot break out of the JS string literal.
- **`unsafe_allow_html` usage.** Custom HTML/CSS is rendered only from
  developer-controlled templates and `styles.css`, not from raw user input.
  Section titles and contents are surfaced via Streamlit primitives (`st.info`,
  `st.dataframe`) that escape their inputs.
- **Local file inputs only.** The app accepts uploads exclusively through
  Streamlit's `st.file_uploader`; no arbitrary file paths are read from disk.

If you fork this project, keep these properties in mind before adding any
remote integrations.

---

## Project Structure

```
.
├── app.py                  # Streamlit UI entry point
├── document_processor.py   # File parsing + section detection
├── semantic_search.py      # Embeddings + similarity search
├── styles.css              # UI styling
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT
└── README.md
```

---

## Future Improvements

- Persistent document index (SQLite + on-disk embeddings) so re-uploading
  isn't required between sessions.
- GPU acceleration when CUDA is available.
- Pluggable embedding backends (OpenAI, Cohere, local LLM).
- Multi-language model option for non-English documents.
- Batch / CLI mode for headless pipelines.
- OCR support for scanned PDFs (Tesseract / `pytesseract`).

---

## License

Released under the [MIT License](LICENSE).
