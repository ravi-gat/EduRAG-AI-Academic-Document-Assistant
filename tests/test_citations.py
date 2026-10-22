"""Tests for citation validation and integrity checking."""

import pytest
from src.config import DocumentChunk, RetrievalResult, Citation
from src.citations import (
    build_context_block,
    validate_and_extract_citations,
    clean_passage_excerpt,
)


@pytest.fixture
def retrieved_passages():
    """Create sample retrieved passages for citation testing."""
    c1 = DocumentChunk(
        chunk_id="crypto.pdf_p12_c0",
        source="crypto.pdf",
        page=12,
        text="RSA is an asymmetric algorithm that uses a public key and a private key.",
        word_count=14,
    )
    c2 = DocumentChunk(
        chunk_id="crypto.pdf_p13_c0",
        source="crypto.pdf",
        page=13,
        text="Euler's totient function phi(n) is calculated as (p-1)*(q-1).",
        word_count=8,
    )
    c3 = DocumentChunk(
        chunk_id="networking.pdf_p7_c0",
        source="networking.pdf",
        page=7,
        text="TCP provides reliable, ordered, and error-checked delivery of a stream of octets.",
        word_count=13,
    )
    return [
        RetrievalResult(chunk=c1, similarity_score=0.88),
        RetrievalResult(chunk=c2, similarity_score=0.75),
        RetrievalResult(chunk=c3, similarity_score=0.45),
    ]


def test_build_context_block(retrieved_passages):
    """Test context block includes explicit source headers with filename and PDF page."""
    context_text, id_map = build_context_block(retrieved_passages)
    assert "[Source 1 | Document: crypto.pdf | PDF Page: 12]" in context_text
    assert "[Source 2 | Document: crypto.pdf | PDF Page: 13]" in context_text
    assert "[Source 3 | Document: networking.pdf | PDF Page: 7]" in context_text
    assert len(id_map) == 3
    assert id_map[1].source == "crypto.pdf"
    assert id_map[1].page == 12


def test_validate_citations_valid_tags(retrieved_passages):
    """Test extracting citations when LLM explicitly cites [Source 1] and [Source 2]."""
    llm_output = (
        "RSA utilizes a public and private key pair [Source 1]. "
        "The security relies on Euler's totient function [Source 2]."
    )
    citations = validate_and_extract_citations(llm_output, retrieved_passages)
    assert len(citations) == 2

    # Verify citation 1
    assert citations[0].source == "crypto.pdf"
    assert citations[0].page == 12
    assert "RSA is an asymmetric" in citations[0].excerpt
    assert citations[0].similarity_score == 0.88

    # Verify citation 2
    assert citations[1].source == "crypto.pdf"
    assert citations[1].page == 13
    assert "Euler's totient" in citations[1].excerpt


def test_reject_hallucinated_citations(retrieved_passages):
    """Ensure nonexistent sources or hallucinated tags (e.g. [Source 99]) are rejected."""
    llm_output = (
        "Quantum computing will break all encryption [Source 99] "
        "according to imaginary_textbook.pdf [Source 8]."
    )
    citations = validate_and_extract_citations(llm_output, retrieved_passages)
    # Hallucinated tags must NOT become citations
    for c in citations:
        assert c.source in ["crypto.pdf", "networking.pdf"]
        assert c.page in [12, 13, 7]


def test_clean_passage_excerpt():
    """Test passage excerpt truncation for clean display."""
    long_passage = " ".join([f"token{i}" for i in range(100)])
    excerpt = clean_passage_excerpt(long_passage, max_words=20)
    assert len(excerpt.split()) == 21  # 20 words + '...'
    assert excerpt.endswith("...")
