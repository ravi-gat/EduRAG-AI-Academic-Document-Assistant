"""EduRAG — AI Academic Document Assistant.

Challenge Theme: "From Text to Insight — Ask Your Documents"
Tagline: "Your Knowledge. Your Documents. Your Answers."

Production-grade Streamlit application implementing:
1. INGEST   - Page-by-page PDF extraction with PyMuPDF
2. RETRIEVE - FAISS vector similarity search with Sentence Transformers
3. GENERATE - Grounded response generation using Google Gen AI SDK (Gemini)
4. CITE     - Exact source document and 1-based PDF page citations with excerpts
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional

import streamlit as st
import numpy as np

from src.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GEMINI_MODEL,
    SUPPORTED_GEMINI_MODELS,
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    INSUFFICIENT_EVIDENCE_RESPONSE,
    DocumentChunk,
    RetrievalResult,
    Citation,
    AnswerResult,
    get_gemini_api_key,
)
from src.pdf_processor import (
    extract_pages_from_bytes,
    extract_pages_from_file,
    PDFExtractionError,
    EmptyPDFError,
    ScannedPDFError,
)
from src.chunking import chunk_pages
from src.embeddings import (
    get_embedding_model,
    generate_embeddings,
    generate_query_embedding,
)
from src.retrieval import FAISSIndexManager
from src.answer_generator import generate_answer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EduRAG")

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EduRAG — AI Academic Document Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Academic CSS Styling
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header branding */
.edurag-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #2563eb 0%, #38bdf8 50%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
    letter-spacing: -0.02em;
}

.edurag-tagline {
    font-size: 1.05rem;
    font-weight: 500;
    color: #64748b;
    margin-bottom: 1.2rem;
}

.hackathon-badge {
    display: inline-block;
    background: rgba(37, 99, 235, 0.08);
    border: 1px solid rgba(37, 99, 235, 0.25);
    color: #1d4ed8;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
}

/* Metric stat cards */
.stat-card {
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(226, 232, 240, 0.9);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
}

.stat-num {
    font-size: 1.8rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
}

.stat-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 0.25rem;
}

/* Citation card */
.citation-container {
    margin-top: 1rem;
    border-top: 1px solid #e2e8f0;
    padding-top: 0.75rem;
}

.citation-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}

.citation-card:hover {
    background: #f1f5f9;
    border-left-color: #2563eb;
}

.citation-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 600;
    font-size: 0.9rem;
    color: #1e293b;
    margin-bottom: 0.4rem;
}

.citation-badge {
    background: #dbeafe;
    color: #1e40af;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.15rem 0.55rem;
    border-radius: 4px;
}

.citation-score {
    background: #e0f2fe;
    color: #0369a1;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    margin-left: 0.5rem;
}

.citation-excerpt {
    font-size: 0.84rem;
    color: #475569;
    line-height: 1.45;
    background: #ffffff;
    padding: 0.6rem;
    border-radius: 6px;
    border: 1px dashed #cbd5e1;
    font-style: italic;
    margin-top: 0.4rem;
}

/* Insufficient evidence warning */
.insufficient-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 1rem;
    color: #92400e;
    font-size: 0.92rem;
    font-weight: 500;
    margin-top: 0.5rem;
}

/* Sidebar document chips */
.doc-chip {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.55rem 0.85rem;
    margin-bottom: 0.45rem;
    font-size: 0.82rem;
}

.doc-name {
    font-weight: 600;
    color: #1e293b;
    max-width: 170px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.doc-meta {
    font-size: 0.72rem;
    color: #64748b;
}

/* Code and mono font */
code {
    font-family: 'JetBrains Mono', monospace !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "retriever" not in st.session_state:
    st.session_state.retriever = FAISSIndexManager()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_doc_records" not in st.session_state:
    st.session_state.uploaded_doc_records = {}  # {filename: {"pages": int, "chunks": int, "words": int}}


# Preload embedding model into cache
@st.cache_resource(show_spinner="Initializing Sentence Transformer model (all-MiniLM-L6-v2)...")
def load_cached_embedding_model():
    return get_embedding_model(DEFAULT_EMBEDDING_MODEL)

try:
    embedding_model = load_cached_embedding_model()
except Exception as e:
    st.error(f"Failed to load embedding model: {e}")
    embedding_model = None


# -----------------------------------------------------------------------------
# Document Processing Helper Functions
# -----------------------------------------------------------------------------
def process_pdf_document(
    file_bytes: bytes,
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
) -> bool:
    """Ingest a PDF from raw bytes, extract text page-by-page, chunk, embed, and index."""
    try:
        pages_data = extract_pages_from_bytes(file_bytes, filename)
        chunks = chunk_pages(pages_data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if not chunks:
            st.warning(f"'{filename}' produced 0 text chunks.")
            return False

        texts = [c.text for c in chunks]
        embeddings = generate_embeddings(texts, model=embedding_model)

        st.session_state.retriever.add_chunks(chunks, embeddings)

        # Record document metadata
        unique_pages = len(pages_data)
        total_words = sum(c.word_count for c in chunks)
        st.session_state.uploaded_doc_records[filename] = {
            "pages": unique_pages,
            "chunks": len(chunks),
            "words": total_words,
        }
        return True

    except ScannedPDFError as spe:
        st.error(f"Scanned PDF Detected: {spe}")
        return False
    except EmptyPDFError as epe:
        st.error(f"Empty Document: {epe}")
        return False
    except PDFExtractionError as pee:
        st.error(f"Extraction Error for '{filename}': {pee}")
        return False
    except Exception as e:
        st.error(f"Unexpected processing error on '{filename}': {e}")
        return False


def load_local_sample_pdf(relative_path: str, chunk_size: int, chunk_overlap: int):
    """Load one of the included educational sample PDFs."""
    if not os.path.exists(relative_path):
        st.error(f"Sample file not found: {relative_path}")
        return

    filename = os.path.basename(relative_path)
    if filename in st.session_state.retriever.document_names:
        st.info(f"'{filename}' is already indexed in the knowledge base.")
        return

    with open(relative_path, "rb") as f:
        file_bytes = f.read()

    with st.spinner(f"Ingesting educational sample: {filename}..."):
        success = process_pdf_document(file_bytes, filename, chunk_size, chunk_overlap)
        if success:
            st.success(f"Indexed sample document: {filename}")
            st.rerun()


# -----------------------------------------------------------------------------
# Sidebar: Knowledge Base & Configuration Management
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 EduRAG Assistant")
    st.markdown(
        "<span class='hackathon-badge'>Challenge: Ask Your Documents</span>",
        unsafe_allow_html=True,
    )
    st.caption("Grounded Academic Document Q&A with Strict Page Citations")

    st.markdown("---")

    # API Key Configuration
    st.markdown("#### 🔑 Model & API Settings")
    env_key = get_gemini_api_key()
    api_key_input = st.text_input(
        "Gemini API Key",
        value=env_key or "",
        type="password",
        placeholder="Enter your Gemini API key...",
        help="Reads from GEMINI_API_KEY environment variable, .env file, or direct input.",
    )

    if api_key_input:
        st.caption("✅ API key configured.")
    else:
        st.caption("⚠️ API key required for live answer generation.")

    # Model Selector
    selected_model = st.selectbox(
        "Gemini Model",
        options=SUPPORTED_GEMINI_MODELS,
        index=0,
        help="Select the Gemini model for synthesis.",
    )

    # Retrieval Tuning Expander
    with st.expander("⚙️ Advanced Retrieval Tuning", expanded=False):
        top_k = st.slider("Top-K Passages", min_value=1, max_value=10, value=DEFAULT_TOP_K)
        similarity_threshold = st.slider(
            "Evidence Similarity Threshold",
            min_value=0.0,
            max_value=0.8,
            value=DEFAULT_SIMILARITY_THRESHOLD,
            step=0.05,
            help="Queries with top passage score below this threshold will trigger insufficient evidence fallback.",
        )
        chunk_size = st.slider(
            "Chunk Size (Words)",
            min_value=100,
            max_value=350,
            value=DEFAULT_CHUNK_SIZE_WORDS,
            step=25,
        )
        chunk_overlap = st.slider(
            "Chunk Overlap (Words)",
            min_value=10,
            max_value=80,
            value=DEFAULT_CHUNK_OVERLAP_WORDS,
            step=5,
        )

    st.markdown("---")

    # Document Ingestion Section
    st.markdown("#### 📂 Ingest Documents")
    uploaded_files = st.file_uploader(
        "Upload Academic PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload lecture notes, textbooks, research papers, or syllabus documents.",
    )

    if uploaded_files:
        if st.button("⚡ Ingest Uploaded Documents", use_container_width=True, type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            for idx, file in enumerate(uploaded_files):
                fname = file.name
                if fname in st.session_state.retriever.document_names:
                    status_text.text(f"Skipping already indexed file: {fname}")
                    continue

                status_text.text(f"Processing ({idx+1}/{len(uploaded_files)}): {fname}...")
                file_bytes = file.read()
                process_pdf_document(file_bytes, fname, chunk_size, chunk_overlap)
                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.text("Ingestion completed!")
            time.sleep(0.5)
            st.rerun()

    # Quick Sample Educational Datasets
    st.markdown("##### 📚 Quick Demo Samples")
    col_sample1, col_sample2 = st.columns(2)
    with col_sample1:
        if st.button("📘 CS Notes", use_container_width=True, help="Load 5-page CS & Algorithms sample"):
            load_local_sample_pdf(
                os.path.join("sample_documents", "computer_science_algorithms.pdf"),
                chunk_size,
                chunk_overlap,
            )
    with col_sample2:
        if st.button("⚛️ Physics", use_container_width=True, help="Load 4-page Quantum Mechanics sample"):
            load_local_sample_pdf(
                os.path.join("sample_documents", "physics_quantum_mechanics.pdf"),
                chunk_size,
                chunk_overlap,
            )

    st.markdown("---")

    # Active Documents Management
    st.markdown("#### 📑 Active Knowledge Base")
    doc_stats = st.session_state.retriever.get_document_stats()

    if not doc_stats:
        st.info("No documents indexed yet. Upload a PDF or load a sample above to begin.")
    else:
        for doc_name, stats in doc_stats.items():
            st.markdown(
                f"""
                <div class="doc-chip">
                    <div>
                        <div class="doc-name" title="{doc_name}">📄 {doc_name}</div>
                        <div class="doc-meta">{stats['pages']} pages • {stats['chunks']} chunks</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_rem, _ = st.columns([1, 2])
            with col_rem:
                if st.button("Remove", key=f"del_{doc_name}", help=f"Remove {doc_name} and rebuild index"):
                    st.session_state.retriever.remove_document(doc_name)
                    if doc_name in st.session_state.uploaded_doc_records:
                        del st.session_state.uploaded_doc_records[doc_name]
                    st.rerun()

        if st.button("🗑️ Clear Entire Knowledge Base", use_container_width=True):
            st.session_state.retriever.clear()
            st.session_state.uploaded_doc_records.clear()
            st.session_state.messages.clear()
            st.rerun()


# -----------------------------------------------------------------------------
# Main Application Workspace
# -----------------------------------------------------------------------------

# Title & Academic Dashboard Header
st.markdown("<div class='edurag-title'>EduRAG</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='edurag-tagline'>Your Knowledge. Your Documents. Your Answers.</div>",
    unsafe_allow_html=True,
)

# Knowledge Base Dashboard Metric Indicators
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

total_docs = st.session_state.retriever.total_documents
total_chunks = st.session_state.retriever.total_chunks
total_pages = sum(s["pages"] for s in st.session_state.retriever.get_document_stats().values())
index_status = "Ready 🟢" if total_chunks > 0 else "Empty ⚪"

with col_m1:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-num">{total_docs}</div>
            <div class="stat-label">Indexed Documents</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m2:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-num">{total_pages}</div>
            <div class="stat-label">Total PDF Pages</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m3:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-num">{total_chunks}</div>
            <div class="stat-label">Vector Chunks (FAISS)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m4:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-num">{index_status}</div>
            <div class="stat-label">Vector Search Status</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# -----------------------------------------------------------------------------
# Sample Questions Carousel / Quick Pills
# -----------------------------------------------------------------------------
suggested_q = None

if total_chunks > 0:
    st.markdown("##### 💡 Suggested Questions from Your Course Materials")
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        if st.button("🔍 What is the time complexity of MergeSort?", use_container_width=True):
            suggested_q = "What is the time complexity of MergeSort?"
    with q_col2:
        if st.button("🔐 How does the RSA algorithm work?", use_container_width=True):
            suggested_q = "How does the RSA algorithm work and what are its key components?"
    with q_col3:
        if st.button("⚛️ Explain Heisenberg's Uncertainty Principle", use_container_width=True):
            suggested_q = "What is the Heisenberg Uncertainty Principle and what is its mathematical formula?"
else:
    st.info(
        "👋 **Welcome to EduRAG!** To get started, upload an educational PDF in the sidebar "
        "or click **'📘 CS Notes'** or **'⚛️ Physics'** to load instant sample textbooks."
    )


# -----------------------------------------------------------------------------
# Chat Workspace & Conversation History
# -----------------------------------------------------------------------------
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Render validated source citations
            citations: List[Citation] = msg.get("citations", [])
            if citations:
                st.markdown("<div class='citation-container'><strong>📚 Grounded Citations:</strong></div>", unsafe_allow_html=True)
                for cit in citations:
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-header">
                                <span>📄 <strong>{cit.source}</strong></span>
                                <div>
                                    <span class="citation-badge">PDF Page {cit.page}</span>
                                    <span class="citation-score">Sim: {cit.similarity_score:.2f}</span>
                                </div>
                            </div>
                            <div class="citation-excerpt">
                                "{cit.excerpt}"
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Technical inspection expander for judges / technical review
            retrieved: List[RetrievalResult] = msg.get("retrieved", [])
            if retrieved:
                with st.expander("🔍 Inspect FAISS Retrieved Passages & Similarity Scores"):
                    for i, r in enumerate(retrieved, 1):
                        st.markdown(
                            f"**Passage {i}** — `{r.source}` (PDF Page {r.page}) | "
                            f"Cosine Similarity: `{r.similarity_score:.4f}`"
                        )
                        st.text(r.text)
                        st.divider()


# -----------------------------------------------------------------------------
# Question Processing Pipeline
# -----------------------------------------------------------------------------
user_question = st.chat_input("Ask a question about your uploaded documents...")

# Support clicking suggested questions
if suggested_q and not user_question:
    user_question = suggested_q

if user_question:
    # 1. Display user query in chat
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # 2. Assistant answering block
    with st.chat_message("assistant"):
        if total_chunks == 0:
            fallback_msg = (
                "Please upload or load at least one educational PDF into the knowledge base before asking questions."
            )
            st.warning(fallback_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": fallback_msg,
                "citations": [],
                "retrieved": [],
            })
        else:
            with st.spinner("Searching document index and synthesizing grounded answer..."):
                # Stage 1: Generate query embedding
                query_vec = generate_query_embedding(user_question, model=embedding_model)

                # Stage 2: Retrieve top-k passages from FAISS
                retrieved_chunks = st.session_state.retriever.search(
                    query_embedding=query_vec,
                    top_k=top_k,
                    threshold=0.0,  # Pass all top-k to generator to evaluate evidence threshold
                )

                # Stage 3: Grounded Answer Generation
                answer_result: AnswerResult = generate_answer(
                    question=user_question,
                    retrieved_chunks=retrieved_chunks,
                    api_key=api_key_input,
                    model_name=selected_model,
                    similarity_threshold=similarity_threshold,
                )

            # Display error if API failed
            if answer_result.error_message:
                st.error(f"⚠️ {answer_result.error_message}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ {answer_result.error_message}",
                    "citations": [],
                    "retrieved": retrieved_chunks,
                })

            elif not answer_result.has_sufficient_evidence:
                st.markdown(
                    f"""
                    <div class="insufficient-box">
                        <strong>⚠️ Insufficient Evidence:</strong><br>
                        {answer_result.answer}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if retrieved_chunks:
                    with st.expander("🔍 Why couldn't this be answered? Inspect closest passages"):
                        top_score = max(r.similarity_score for r in retrieved_chunks)
                        st.caption(
                            f"Top retrieved passage had a similarity score of **{top_score:.4f}**, "
                            f"which is below your required evidence threshold (**{similarity_threshold:.4f}**)."
                        )
                        for r in retrieved_chunks[:2]:
                            st.markdown(f"**{r.source}** (PDF Page {r.page}):")
                            st.text(r.text[:200] + "...")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_result.answer,
                    "citations": [],
                    "retrieved": retrieved_chunks,
                })

            else:
                # Render answer markdown
                st.markdown(answer_result.answer)

                # Render Citation Cards
                if answer_result.citations:
                    st.markdown("<div class='citation-container'><strong>📚 Grounded Citations:</strong></div>", unsafe_allow_html=True)
                    for cit in answer_result.citations:
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <div class="citation-header">
                                    <span>📄 <strong>{cit.source}</strong></span>
                                    <div>
                                        <span class="citation-badge">PDF Page {cit.page}</span>
                                        <span class="citation-score">Sim: {cit.similarity_score:.2f}</span>
                                    </div>
                                </div>
                                <div class="citation-excerpt">
                                    "{cit.excerpt}"
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # Inspection Expander
                with st.expander("🔍 Inspect FAISS Retrieved Passages & Similarity Scores"):
                    for i, r in enumerate(retrieved_chunks, 1):
                        st.markdown(
                            f"**Passage {i}** — `{r.source}` (PDF Page {r.page}) | "
                            f"Cosine Similarity: `{r.similarity_score:.4f}`"
                        )
                        st.text(r.text)
                        st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_result.answer,
                    "citations": answer_result.citations,
                    "retrieved": retrieved_chunks,
                })


# Footer & Controls
st.markdown("---")
col_foot1, col_foot2 = st.columns([4, 1])
with col_foot1:
    st.caption("EduRAG • Hackathon Challenge: From Text to Insight — Ask Your Documents • Built with PyMuPDF, Sentence Transformers, FAISS, and Gemini.")
with col_foot2:
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages.clear()
        st.rerun()
