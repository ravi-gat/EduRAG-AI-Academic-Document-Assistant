"""Tests for text chunking and metadata preservation."""

import pytest
from src.chunking import (
    split_text_into_chunks,
    generate_chunk_id,
    chunk_pages,
)
from src.config import DocumentChunk


def test_split_text_into_chunks_basic():
    """Test splitting a passage into chunks of specified size and overlap."""
    words = [f"word{i}" for i in range(100)]
    text = " ".join(words)

    # 40 words chunk, 10 overlap -> step = 30
    chunks = split_text_into_chunks(text, chunk_size=40, chunk_overlap=10)
    assert len(chunks) == 3

    # Check words in chunk 1 and chunk 2 for overlap
    c1_words = chunks[0].split()
    c2_words = chunks[1].split()

    assert len(c1_words) == 40
    # Last 10 words of c1 should equal first 10 words of c2
    assert c1_words[-10:] == c2_words[:10]


def test_split_short_text():
    """Text shorter than chunk_size should produce exactly one chunk."""
    text = "Quantum computing harnesses the phenomena of superposition and entanglement."
    chunks = split_text_into_chunks(text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_empty_text():
    """Empty or whitespace text should produce empty list."""
    assert split_text_into_chunks("") == []
    assert split_text_into_chunks("   \n\t  ") == []


def test_overlap_safety():
    """When overlap >= chunk_size, fallback should prevent infinite loop."""
    text = " ".join([f"item{i}" for i in range(50)])
    chunks = split_text_into_chunks(text, chunk_size=20, chunk_overlap=30)
    assert len(chunks) > 0


def test_generate_chunk_id():
    """Test deterministic and sanitized chunk ID generation."""
    cid = generate_chunk_id("math textbook (v1).pdf", page=7, chunk_idx=2)
    assert "math_textbook__v1_.pdf_p7_c2" in cid
    assert cid == "math_textbook__v1_.pdf_p7_c2"


def test_chunk_pages_preserves_metadata():
    """Test that chunk_pages attaches accurate source, page, and chunk_id metadata."""
    sample_pages = [
        {
            "source": "intro_cs.pdf",
            "page": 1,
            "text": " ".join([f"p1_word{i}" for i in range(60)]),
        },
        {
            "source": "intro_cs.pdf",
            "page": 2,
            "text": " ".join([f"p2_word{i}" for i in range(60)]),
        },
    ]

    chunks = chunk_pages(sample_pages, chunk_size=30, chunk_overlap=10)
    assert len(chunks) >= 4

    for chunk in chunks:
        assert isinstance(chunk, DocumentChunk)
        assert chunk.source == "intro_cs.pdf"
        assert chunk.page in (1, 2)
        assert chunk.chunk_id.startswith("intro_cs.pdf_p")
        assert chunk.word_count > 0


def test_chunk_pages_skips_empty():
    """Empty pages should be safely skipped."""
    pages = [
        {"source": "empty.pdf", "page": 1, "text": ""},
        {"source": "empty.pdf", "page": 2, "text": "   "},
        {"source": "empty.pdf", "page": 3, "text": "Real content here."},
    ]
    chunks = chunk_pages(pages, chunk_size=20, chunk_overlap=5)
    assert len(chunks) == 1
    assert chunks[0].page == 3
    assert chunks[0].text == "Real content here."
