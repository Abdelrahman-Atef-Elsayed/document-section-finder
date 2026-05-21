"""Document Section Finder — Streamlit application.

Upload PDF / DOCX / TXT files, then run semantic queries against their
sections to retrieve the relevant section numbers (e.g. 3.7, 0.6.5.1).
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from document_processor import (
    build_sections_dataframe,
    detect_sections,
    missing_optional_libraries,
    parse_document,
)
from semantic_search import (
    encode_corpus,
    format_section_references,
    load_model,
    search,
    similarity_badge,
)

BASE_DIR = Path(__file__).parent
STYLES_PATH = BASE_DIR / "styles.css"

EXAMPLE_QUERIES = (
    "Example: What are the requirements for user authentication?\n"
    "Example: Data retention and privacy policies\n"
    "Example: Compliance procedures for security audits"
)


def configure_page() -> None:
    st.set_page_config(
        page_title="Document Section Finder",
        page_icon="📄",
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
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_resource(show_spinner="Loading AI model...")
def get_model():
    return load_model()


def render_header() -> None:
    st.markdown(
        """
        <div class="main-header">
            <h1>📄 Document Section Finder</h1>
            <p>Upload documents and search for relevant sections — results look like: 3.7, 3.5, 2.4, 0.1.1, 0.6.5.1</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[int, float]:
    with st.sidebar:
        st.markdown("### ⚙️ Search Settings")
        top_k = st.slider("📊 Number of results", min_value=1, max_value=30, value=10)
        min_similarity = st.slider(
            "🎯 Minimum similarity", min_value=0.1, max_value=1.0, value=0.3, step=0.05
        )

        st.markdown("---")
        st.markdown("### 📈 Document Statistics")
        df = st.session_state.sections_df
        if df is not None and not df.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Sections", f"{len(df):,}")
            with col2:
                st.metric("Documents", df["source"].nunique())
        else:
            st.info("No documents loaded yet")

        st.markdown("---")
        st.markdown("### 📝 Search History")
        if st.session_state.search_history:
            for query in reversed(st.session_state.search_history[-5:]):
                st.caption(f"🔍 {query[:40]}...")
        else:
            st.info("No search history yet")

        if st.session_state.search_history and st.button("🗑️ Clear History"):
            st.session_state.search_history = []
            st.rerun()

        st.markdown("---")
        if st.button("🔄 Clear All Data", help="Clear all loaded documents and start fresh"):
            st.session_state.documents = {}
            st.session_state.sections_df = None
            st.session_state.embeddings = None
            st.success("All data cleared!")
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
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ {uploaded_file.name}: failed to parse ({exc})")
                continue

            if not text:
                st.warning(f"⚠️ Could not extract text from {uploaded_file.name}")
                continue

            sections = detect_sections(text, uploaded_file.name)
            all_sections.extend(sections)
            st.success(f"✅ Extracted {len(sections)} sections from {uploaded_file.name}")
    return all_sections


def render_upload_tab(model) -> None:
    st.markdown("### Upload Your Documents")
    st.markdown("Supported formats: **PDF**, **DOCX (Word)**, **TXT**")

    uploaded_files = st.file_uploader(
        "Drop files here or click to upload",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="You can upload multiple files at once",
    )

    if uploaded_files:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"📁 {len(uploaded_files)} file(s) selected")
        with col2:
            process_clicked = st.button("🚀 Process Files", type="primary", use_container_width=True)

        if process_clicked:
            sections = process_uploaded_files(uploaded_files)
            if sections:
                df = build_sections_dataframe(sections)
                st.session_state.sections_df = df
                with st.spinner("Creating AI embeddings..."):
                    st.session_state.embeddings = encode_corpus(model, df)
                st.success(
                    f"✅ Successfully processed {len(sections)} sections from {len(uploaded_files)} document(s)"
                )
                st.markdown("#### Preview of Extracted Sections")
                st.dataframe(df[["number", "title", "source"]].head(10), use_container_width=True)
            else:
                st.error("❌ No sections could be extracted from the uploaded files.")

    df = st.session_state.sections_df
    if df is not None and not df.empty:
        st.markdown("---")
        st.markdown("### 📊 Currently Loaded Data")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"<div class='stat-card'><h3>{len(df):,}</h3><p>Total Sections</p></div>",
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
        st.warning("⚠️ Please upload and process documents first in the 'Upload Documents' tab.")
        return

    st.markdown("### 🔍 Search for Relevant Sections")
    query = st.text_area(
        "Enter your search query:",
        height=120,
        placeholder=EXAMPLE_QUERIES,
        key="search_input",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.rerun()

    if not search_clicked:
        return

    if not query.strip():
        st.warning("⚠️ Please enter a search query.")
        return

    if query not in st.session_state.search_history:
        st.session_state.search_history.append(query)

    with st.spinner("Searching..."):
        results, section_numbers = search(
            model,
            query,
            df,
            st.session_state.embeddings,
            top_k=top_k,
            min_similarity=min_similarity,
        )

    if not results:
        st.error("❌ No matching sections found. Try lowering the minimum similarity or adjusting your query.")
        return

    refs_formatted = format_section_references(section_numbers)
    render_results(results, refs_formatted)


def render_results(results, refs_formatted: str) -> None:
    ref_col1, ref_col2 = st.columns([5, 1])

    with ref_col1:
        st.markdown(
            f"""
            <div class="reference-output">
                <h3>📚 Matching Section References ({len(results)} found)</h3>
                <p class="refs">{refs_formatted}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ref_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        # json.dumps safely escapes quotes, backticks not part of JSON strings,
        # and prevents breaking out of the JS string literal.
        safe_refs = json.dumps(refs_formatted)
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
        title_preview = res.title[:60] + ("..." if len(res.title) > 60 else "")
        with st.expander(f"**{res.number}** - {title_preview} | {badge}", expanded=(i <= 3)):
            st.markdown(
                f"""
                <div class="section-preview">
                    <h4>📑 Section {res.number}: {res.title}</h4>
                    <p><strong>Source:</strong> {res.source}</p>
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


def render_sections_tab() -> None:
    df = st.session_state.sections_df
    if df is None or df.empty:
        st.warning("⚠️ Please upload and process documents first in the 'Upload Documents' tab.")
        return

    st.markdown("### 📋 All Extracted Sections")
    docs = ["All Documents"] + list(df["source"].unique())
    selected = st.selectbox("Filter by document:", docs)

    display_df = df if selected == "All Documents" else df[df["source"] == selected]
    st.info(f"Showing {len(display_df)} sections")
    st.dataframe(display_df[["number", "title", "source"]], use_container_width=True, height=500)

    if st.button("📥 Download Sections as CSV"):
        csv = display_df[["number", "title", "content", "source"]].to_csv(index=False)
        st.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name="extracted_sections.csv",
            mime="text/csv",
        )


def render_footer() -> None:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="footer-container">
            <div style='text-align: center; color: #6b7280;'>
                <p style='font-size: 1.1rem; font-weight: 600; margin-bottom: 0.75rem;'>
                    📄 Document Section Finder
                </p>
                <p style='font-size: 0.95rem; margin-bottom: 0.5rem;'>
                    Powered by Sentence Transformers &amp; Streamlit
                </p>
                <p style='font-size: 0.9rem;'>
                    Supports PDF, DOCX, and TXT formats | Semantic search with AI
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

    model = get_model()

    render_header()
    top_k, min_similarity = render_sidebar()

    missing = missing_optional_libraries()
    if missing:
        st.warning(f"⚠️ Some libraries are not installed: {', '.join(missing)}. Install them for full functionality.")

    tab_upload, tab_search, tab_view = st.tabs(["📤 Upload Documents", "🔍 Search", "📋 View Sections"])
    with tab_upload:
        render_upload_tab(model)
    with tab_search:
        render_search_tab(model, top_k, min_similarity)
    with tab_view:
        render_sections_tab()

    render_footer()


if __name__ == "__main__":
    main()
