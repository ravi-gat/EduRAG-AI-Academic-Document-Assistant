"""Configuration and shared data structures for EduRAG."""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

# Embedding Model Configuration
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # MiniLM-L6-v2 produces 384-dimensional embeddings

# Generation Model Configuration
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
SUPPORTED_GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.8-pro",
]

# Chunking Configuration
DEFAULT_CHUNK_SIZE_WORDS = 200      # 150 - 250 words per chunk
DEFAULT_CHUNK_OVERLAP_WORDS = 40    # 30 - 50 words overlap

# Retrieval Configuration
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.35  # Cosine similarity calibration threshold
MAX_FILE_SIZE_MB = 50

# Fallback & Grounding Prompts
INSUFFICIENT_EVIDENCE_RESPONSE = (
    "I don't know based on the provided documents. "
    "Please upload a relevant document or ask a question supported by your current documents."
)

SYSTEM_INSTRUCTION = (
    "You are EduRAG, an academic document assistant. Answer using the supplied document passages. "
    "Do not invent facts, quotations, page numbers, or source references. "
    "If the supplied passages do not contain sufficient evidence, clearly state that the answer cannot "
    "be determined from the uploaded documents."
)


def get_gemini_api_key() -> Optional[str]:
    """Retrieve Gemini API key from environment or Streamlit secrets."""
    # 1. Environment variables
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key and key.strip():
        return key.strip()

    # 2. Streamlit secrets (if running inside Streamlit runtime)
    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets:
                return str(st.secrets["GEMINI_API_KEY"]).strip()
            if "GOOGLE_API_KEY" in st.secrets:
                return str(st.secrets["GOOGLE_API_KEY"]).strip()
    except Exception:
        pass

    return None


@dataclass
class DocumentChunk:
    """Represents an extracted and chunked passage from an educational document."""
    chunk_id: str
    source: str
    page: int  # 1-based PDF page number
    text: str
    word_count: int = 0
    token_count: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.page,
            "text": self.text,
            "word_count": self.word_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentChunk":
        return cls(
            chunk_id=data["chunk_id"],
            source=data["source"],
            page=data["page"],
            text=data["text"],
            word_count=data.get("word_count", len(data["text"].split())),
        )


@dataclass
class RetrievalResult:
    """Represents a retrieved chunk with its similarity score."""
    chunk: DocumentChunk
    similarity_score: float

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def page(self) -> int:
        return self.chunk.page

    @property
    def text(self) -> str:
        return self.chunk.text


@dataclass
class Citation:
    """Validated source citation for an answer."""
    source: str
    page: int  # 1-based PDF page number
    excerpt: str
    similarity_score: float
    chunk_id: str


@dataclass
class AnswerResult:
    """Full generation result including citations and confidence status."""
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    has_sufficient_evidence: bool = True
    retrieved_chunks: List[RetrievalResult] = field(default_factory=list)
    model_used: str = DEFAULT_GEMINI_MODEL
    error_message: Optional[str] = None
