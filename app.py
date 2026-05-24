from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from document_processor import (
    build_sections_dataframe,
    missing_optional_libraries,
    parse_document,
    split_paragraphs,
    OCR_AVAILABLE,
)
from semantic_search import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL_NAME,
    encode_corpus,
    format_section_references,
    load_model,
    search,
    similarity_badge,
    BM25_AVAILABLE,
    CROSS_ENCODER_AVAILABLE,
)
from vector_index import (
    CHROMA_AVAILABLE,
    index_exists,
    get_document_count,
    store_embeddings,
    load_index_df,
    get_collection_embeddings,
    clear_index,
)
from rag_engine import RAG_AVAILABLE, generate_answer
from export_utils import PDF_EXPORT_AVAILABLE, export_results_pdf

BASE_DIR = Path(__file__).parent
STYLES_PATH = BASE_DIR / "styles.css"

EXAMPLE_QUERIES = (
    "Example: What are the requirements for user authentication?\n"
    "Example: Data retention and privacy policies\n"
    "Example: System architecture and design overview"
)


def configure_page() -> None:
    st.set_page_config(
        page_title="Document Section Finder",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_styles() -> None:
    if STYLES_PATH.exists():
        st.markdown(f"<style>{STYLES_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def init_session_state() -> None:
    defaults = {
        "search_history": [],
        "documents": {},
        "sections_df": None,
        "embeddings": None,
        "model_name": DEFAULT_MODEL_NAME,
        "use_hybrid": True,
        "use_cross_encoder": False,
        "rag_results": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_resource(show_spinner="Loading AI model...")
def get_model(model_name: str):
    return load_model(model_name)


def render_keyboard_shortcuts_js() -> None:
    components.html(
        """
        <script>
        document.addEventListener("keydown", function(e) {
            if (e.key === "Enter" && e.ctrlKey) {
                const btns = document.querySelectorAll('button[kind="primary"]');
                if (btns.length > 0) btns[0].click();
            }
            if (e.key === "Escape") {
                const textareas = document.querySelectorAll('textarea');
                if (textareas.length > 0) textareas[0].value = "";
            }
        });
        </script>
        """,
        height=0,
    )


def render_header() -> None:
    st.markdown(
        f"""
        <div class="main-header">
            <div class="header-content">
                <h1>📄 Document Section Finder</h1>
                <p>Upload any document (PDF, DOCX, TXT) and search with AI — hybrid search, cross-encoder re-ranking, and AI-powered answers</p>
            </div>
            <div class="header-badge">
                <span class="version-badge">v2.0</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple:
    with st.sidebar:
        st.markdown("### 🧠 Model Settings")

        model_names = list(AVAILABLE_MODELS.keys())
        selected_model = st.selectbox(
            "Embedding model",
            model_names,
            index=model_names.index(st.session_state.model_name) if st.session_state.model_name in model_names else 0,
            format_func=lambda x: f"{x} ({AVAILABLE_MODELS[x]})",
            help="Larger models give better quality but use more RAM",
        )
        if selected_model != st.session_state.model_name:
            st.session_state.model_name = selected_model
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("### 🔍 Search Settings")
        top_k = st.slider("📊 Number of results", min_value=1, max_value=30, value=10)
        min_similarity = st.slider("🎯 Minimum similarity", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        hybrid_available = BM25_AVAILABLE
        use_hybrid = st.toggle(
            "Hybrid search (BM25 + semantic)",
            value=st.session_state.use_hybrid and hybrid_available,
            disabled=not hybrid_available,
            help="Combines keyword matching with semantic search for better results",
        )
        st.session_state.use_hybrid = use_hybrid

        ce_available = CROSS_ENCODER_AVAILABLE
        use_ce = st.toggle(
            "Cross-encoder re-ranking",
            value=st.session_state.use_cross_encoder and ce_available,
            disabled=not ce_available,
            help="Re-ranks top results with a precision cross-encoder (slower but more accurate)",
        )
        st.session_state.use_cross_encoder = use_ce

        st.markdown("---")
        st.markdown("### 🤖 AI Answer")
        rag_available = RAG_AVAILABLE
        use_rag = st.toggle(
            "Generate AI answer (RAG)",
            value=False,
            disabled=not rag_available,
            help="Generate a direct answer using a local AI model (flan-t5-large, ~800MB download first run)",
        )
        st.session_state.use_rag = use_rag

        st.markdown("---")
        st.markdown("### 📈 Statistics")
        df = st.session_state.sections_df
        if df is not None and not df.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Paragraphs", f"{len(df):,}")
            with col2:
                st.metric("Documents", df["source"].nunique())
        else:
            st.info("No documents loaded")

        st.markdown("---")
        st.markdown("### 📝 History")
        if st.session_state.search_history:
            for q in reversed(st.session_state.search_history[-5:]):
                st.caption(f"🔍 {q[:40]}...")
        else:
            st.info("No searches yet")

        if st.session_state.search_history and st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.search_history = []
            st.rerun()

        st.markdown("---")
        if st.button("🔄 Reset All", use_container_width=True, type="secondary"):
            st.session_state.sections_df = None
            st.session_state.embeddings = None
            st.session_state.search_history = []
            st.session_state.rag_results = None
            if CHROMA_AVAILABLE and index_exists():
                clear_index()
            st.rerun()

    return top_k, min_similarity


def process_uploaded_files(uploaded_files) -> list:
    all_sections = []
    for uploaded_file in uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            uploaded_file.seek(0)
            try:
                text = parse_document(uploaded_file)
            except (RuntimeError, ValueError) as exc:
                st.error(f"❌ {uploaded_file.name}: {exc}")
                continue
            except Exception as exc:
                st.error(f"❌ {uploaded_file.name}: failed to parse ({exc})")
                continue

            if not text:
                st.warning(f"⚠️ Could not extract text from {uploaded_file.name}")
                continue

            sections = split_paragraphs(text, uploaded_file.name)
            all_sections.extend(sections)
            st.success(f"✅ Extracted {len(sections)} paragraphs from {uploaded_file.name}")
    return all_sections


def render_upload_tab(model) -> None:
    st.markdown("### 📤 Upload Documents")
    st.markdown("Supported formats: **PDF**, **DOCX (Word)**, **TXT**")
    if OCR_AVAILABLE:
        st.info("🔍 OCR is enabled — scanned PDFs will be processed automatically")

    uploaded_files = st.file_uploader(
        "Drop files here or click to upload — drag multiple files at once",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="You can upload multiple files at once",
    )

    col_import, col_persist, col_status = st.columns([2, 1, 1])

    with col_import:
        process_clicked = False
        if uploaded_files:
            if st.button("🚀 Process Files", type="primary", use_container_width=True):
                process_clicked = True

    with col_persist:
        if CHROMA_AVAILABLE:
            persist_label = "💾 Save to Index" if st.session_state.sections_df is not None else None
            if persist_label:
                if st.button(persist_label, use_container_width=True):
                    with st.spinner("Saving to persistent index..."):
                        df = st.session_state.sections_df
                        emb = st.session_state.embeddings
                        if df is not None and emb is not None:
                            emb_list = emb.cpu().numpy().tolist() if hasattr(emb, 'cpu') else emb
                            success = store_embeddings(df, emb_list)
                            if success:
                                st.success("✅ Saved to persistent index")
                            else:
                                st.error("❌ Failed to save index")
                    st.rerun()

                if index_exists():
                    st.caption(f"📦 Index: {get_document_count()} paragraphs stored")

    with col_status:
        if CHROMA_AVAILABLE and index_exists() and st.session_state.sections_df is None:
            if st.button("📂 Load Saved Index", use_container_width=True):
                with st.spinner("Loading persisted index..."):
                    df = load_index_df()
                    if df is not None:
                        st.session_state.sections_df = df
                        emb_raw = get_collection_embeddings()
                        if emb_raw:
                            import torch
                            st.session_state.embeddings = torch.tensor(emb_raw)
                        st.success(f"✅ Loaded {len(df)} paragraphs from disk")
                        st.rerun()
                    else:
                        st.error("❌ No saved index found")

    if process_clicked:
        sections = process_uploaded_files(uploaded_files)
        if sections:
            df = build_sections_dataframe(sections)
            st.session_state.sections_df = df
            with st.spinner("Creating AI embeddings..."):
                st.session_state.embeddings = encode_corpus(model, df)
            st.success(f"✅ Processed {len(sections)} paragraphs from {len(uploaded_files)} document(s)")
            st.markdown("#### Preview")
            st.dataframe(df[["number", "title", "source", "page_number"]].head(10), use_container_width=True)
        else:
            st.error("❌ No paragraphs could be extracted.")

    df = st.session_state.sections_df
    if df is not None and not df.empty:
        st.markdown("---")
        st.markdown("### 📊 Currently Loaded Data")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"<div class='stat-card'><h3>{len(df):,}</h3><p>Paragraphs</p></div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"<div class='stat-card'><h3>{df['source'].nunique()}</h3><p>Documents</p></div>",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"<div class='stat-card'><h3>{len(st.session_state.search_history)}</h3><p>Searches</p></div>",
                unsafe_allow_html=True,
            )


def render_search_tab(model, top_k: int, min_similarity: float) -> None:
    df = st.session_state.sections_df
    if df is None or df.empty:
        st.warning("⚠️ Please upload and process documents first.")
        return

    st.markdown("### 🔍 Search")

    query = st.text_area(
        "Enter your query (Ctrl+Enter to search, Esc to clear):",
        height=100,
        placeholder=EXAMPLE_QUERIES,
        key="search_input",
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.rag_results = None
            st.rerun()

    if not search_clicked:
        return

    if not query.strip():
        st.warning("⚠️ Please enter a search query.")
        return

    if query not in st.session_state.search_history:
        st.session_state.search_history.append(query)

    with st.spinner("Searching..."):
        results, numbers = search(
            model,
            query,
            df,
            st.session_state.embeddings,
            top_k=top_k,
            min_similarity=min_similarity,
            use_hybrid=st.session_state.use_hybrid,
            use_cross_encoder=st.session_state.use_cross_encoder,
        )

    if not results:
        st.error("❌ No matching paragraphs found. Try lowering the minimum similarity.")
        return

    refs = format_section_references(numbers)

    if st.session_state.get("use_rag") and RAG_AVAILABLE:
        with st.spinner("Generating AI answer..."):
            rag_result = generate_answer(query, results)
        if rag_result:
            st.session_state.rag_results = rag_result
        else:
            st.warning("⚠️ Could not generate AI answer (model not downloaded yet? First run downloads ~800MB)")

    rag_result = st.session_state.rag_results
    if rag_result and search_clicked:
        st.markdown("### 🤖 AI Answer")
        with st.container():
            st.markdown(
                f"""
                <div class="rag-answer">
                    <div class="rag-answer-text">{rag_result.answer}</div>
                    <div class="rag-citations">
                        <strong>Sources:</strong> {', '.join(f'[{i}]' for i in rag_result.citations) if rag_result.citations else 'N/A'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_results(results, refs, query)

    if PDF_EXPORT_AVAILABLE:
        st.markdown("---")
        if st.button("📥 Export Results as PDF", use_container_width=True):
            pdf_bytes = export_results_pdf(query, results)
            if pdf_bytes:
                st.download_button(
                    label="💾 Download PDF Report",
                    data=pdf_bytes,
                    file_name="search_results.pdf",
                    mime="application/pdf",
                )
            else:
                st.error("❌ Failed to generate PDF")


def render_results(results, refs_str: str, query: str = "") -> None:
    ref_col1, ref_col2 = st.columns([5, 1])

    with ref_col1:
        st.markdown(
            f"""
            <div class="reference-output">
                <h3>📚 Matching Paragraphs ({len(results)} found)</h3>
                <p class="refs">{refs_str}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ref_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        safe_refs = json.dumps(refs_str)
        components.html(
            f"""
            <div style="padding-top: 20px;">
                <button id="copyRefsBtn"
                    style="
                        padding: 12px 24px;
                        background: linear-gradient(135deg, #eab308 0%, #d97706 100%);
                        color: #fff;
                        border: none;
                        border-radius: 10px;
                        cursor: pointer;
                        font-weight: 600;
                        font-size: 1rem;
                        transition: all 0.3s;
                        box-shadow: 0 4px 6px rgba(234, 179, 8, 0.3);
                        width: 100%;
                    "
                    onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 12px rgba(234, 179, 8, 0.4)';"
                    onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 6px rgba(234, 179, 8, 0.3)';"
                >
                    📋 Copy
                </button>
            </div>
            <script>
            const copyBtn = document.getElementById("copyRefsBtn");
            const refsText = {safe_refs};
            copyBtn.addEventListener("click", () => {{
                navigator.clipboard.writeText(refsText);
                copyBtn.innerHTML = "✅ Copied!";
                setTimeout(() => {{ copyBtn.innerHTML = "📋 Copy"; }}, 2000);
            }});
            </script>
            """,
            height=100,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📄 Detailed Results")

    for i, res in enumerate(results, 1):
        badge = similarity_badge(res.similarity)
        title_preview = res.title[:80] + ("..." if len(res.title) > 80 else "")
        page_info = f" | 📄 Page {res.page_number}" if res.page_number else ""
        expander_title = f"**#{res.number}**{page_info} | {badge}"
        if res.score_bm25 or res.score_rerank:
            extra = ""
            if res.score_bm25:
                extra += f" BM25:{res.score_bm25:.2f}"
            if res.score_rerank:
                extra += f" CE:{res.score_rerank:.2f}"
            expander_title += f" | `{extra.strip()}`"
        with st.expander(expander_title, expanded=(i <= 3)):
            st.markdown(
                f"""
                <div class="paragraph-preview">
                    <h4>📑 Paragraph {res.number}: {title_preview}</h4>
                    <p><strong>Source:</strong> {res.source}{page_info}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if res.snippet_html:
                st.markdown("**Best Match:**")
                st.markdown(
                    f"<div class='snippet-box'>{res.snippet_html}</div>",
                    unsafe_allow_html=True,
                )
            elif res.content and res.content != "N/A":
                st.markdown("**Content Preview:**")
                preview = res.content[:500] + ("..." if len(res.content) > 500 else "")
                st.info(preview)

            # Show source document link
            if hasattr(res, 'source') and res.source:
                raw_html = f"<p style='font-size:0.8em;color:#6b7280;margin-top:8px;'>📁 {res.source}{page_info}</p>"
                st.markdown(raw_html, unsafe_allow_html=True)


def render_paragraphs_tab() -> None:
    df = st.session_state.sections_df
    if df is None or df.empty:
        st.warning("⚠️ Please upload documents first.")
        return

    st.markdown("### 📋 All Extracted Paragraphs")
    docs = ["All Documents"] + list(df["source"].unique())
    selected = st.selectbox("Filter by document:", docs)

    display_df = df if selected == "All Documents" else df[df["source"] == selected]
    st.info(f"Showing {len(display_df)} paragraphs")
    cols = ["number", "title", "source", "page_number"]
    st.dataframe(display_df[cols], use_container_width=True, height=500)

    if st.button("📥 Download as CSV"):
        csv = display_df[["number", "title", "content", "source", "page_number"]].to_csv(index=False)
        st.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name="paragraphs.csv",
            mime="text/csv",
        )


def render_footer() -> None:
    st.markdown("<br><br>", unsafe_allow_html=True)
    rag_status = "✅ RAG ready" if RAG_AVAILABLE else "⬜ RAG (install transformers)"
    index_status = "✅ Index" if CHROMA_AVAILABLE else "⬜ ChromaDB"
    ocr_status = "✅ OCR" if OCR_AVAILABLE else "⬜ OCR (install pytesseract)"
    bm25_status = "✅ BM25" if BM25_AVAILABLE else "⬜ BM25"
    ce_status = "✅ Cross-encoder" if CROSS_ENCODER_AVAILABLE else "⬜ Cross-encoder"
    export_status = "✅ PDF export" if PDF_EXPORT_AVAILABLE else "⬜ PDF export"

    st.markdown(
        f"""
        <div class="footer-container">
            <div style='text-align: center; color: #6b7280;'>
                <p style='font-size: 1.1rem; font-weight: 600; margin-bottom: 0.75rem;'>
                    📄 Document Section Finder
                </p>
                <p style='font-size: 0.85rem; margin-bottom: 0.5rem;'>
                    {bm25_status} | {ce_status} | {rag_status} | {index_status} | {ocr_status} | {export_status}
                </p>
                <p style='font-size: 0.9rem;'>
                    Supports PDF, DOCX, TXT | Hybrid search | Cross-encoder re-ranking | AI answers | Persistent index
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    configure_page()
    inject_styles()
    init_session_state()
    render_keyboard_shortcuts_js()

    model = get_model(st.session_state.model_name)

    render_header()
    top_k, min_similarity = render_sidebar()

    missing = missing_optional_libraries()
    if missing:
        plural = "are" if len(missing) > 1 else "is"
        st.warning(f"⚠️ Optional libraries not installed: {', '.join(missing)}. Install for full functionality.")

    tab_upload, tab_search, tab_view = st.tabs(["📤 Upload", "🔍 Search", "📋 Paragraphs"])
    with tab_upload:
        render_upload_tab(model)
    with tab_search:
        render_search_tab(model, top_k, min_similarity)
    with tab_view:
        render_paragraphs_tab()

    render_footer()


if __name__ == "__main__":
    main()
