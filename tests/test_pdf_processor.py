"""Tests for PDF text extraction module."""

import os
import pytest
import pymupdf as fitz

from src.pdf_processor import (
    extract_pages_from_bytes,
    extract_pages_from_file,
    normalize_text,
    PDFExtractionError,
    EmptyPDFError,
    ScannedPDFError,
)


@pytest.fixture
def sample_pdf_bytes():
    """Create a minimal 2-page in-memory PDF for unit testing."""
    doc = fitz.open()
    # Page 1
    p1 = doc.new_page()
    p1.insert_text(fitz.Point(50, 100), "This is page one discussing Asymptotic Big-O complexity.")
    # Page 2
    p2 = doc.new_page()
    p2.insert_text(fitz.Point(50, 100), "This is page two explaining the RSA public-key cryptosystem.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def empty_pdf_bytes():
    """Create a 1-page PDF with no text (simulating blank / scanned page)."""
    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_normalize_text():
    """Test text cleanup and whitespace normalization."""
    raw = "Algorithm  analysis   and\xa0complexity.\n\n\n\nNext  paragraph."
    normalized = normalize_text(raw)
    assert "  " not in normalized
    assert "\xa0" not in normalized
    assert normalized == "Algorithm analysis and complexity.\n\nNext paragraph."


def test_extract_pages_valid(sample_pdf_bytes):
    """Test successful page-by-page extraction from valid PDF bytes."""
    pages = extract_pages_from_bytes(sample_pdf_bytes, filename="algorithms.pdf")
    assert len(pages) == 2
    assert pages[0]["page"] == 1
    assert pages[0]["source"] == "algorithms.pdf"
    assert "Asymptotic Big-O" in pages[0]["text"]
    assert pages[1]["page"] == 2
    assert "RSA public-key" in pages[1]["text"]


def test_extract_pages_preserves_page_numbers(sample_pdf_bytes):
    """Ensure page numbers are 1-based and strictly sequential."""
    pages = extract_pages_from_bytes(sample_pdf_bytes, filename="test.pdf")
    page_numbers = [p["page"] for p in pages]
    assert page_numbers == [1, 2]


def test_extract_empty_bytes():
    """Test that zero-byte input raises EmptyPDFError."""
    with pytest.raises(EmptyPDFError):
        extract_pages_from_bytes(b"", filename="empty.pdf")


def test_extract_corrupted_bytes():
    """Test that malformed/corrupted bytes raise PDFExtractionError."""
    with pytest.raises(PDFExtractionError):
        extract_pages_from_bytes(b"Not a valid PDF header content", filename="corrupt.pdf")


def test_extract_scanned_pdf(empty_pdf_bytes):
    """Test that PDFs without digital text raise ScannedPDFError requiring OCR."""
    with pytest.raises(ScannedPDFError):
        extract_pages_from_bytes(empty_pdf_bytes, filename="scanned.pdf")


def test_extract_from_file_existing():
    """Test extraction from sample_documents directory if generated."""
    sample_file = os.path.join("sample_documents", "computer_science_algorithms.pdf")
    if os.path.exists(sample_file):
        pages = extract_pages_from_file(sample_file)
        assert len(pages) == 5
        assert pages[0]["page"] == 1
        assert "Algorithm Analysis" in pages[0]["text"]


def test_extract_from_file_nonexistent():
    """Test that non-existent file path raises PDFExtractionError."""
    with pytest.raises(PDFExtractionError):
        extract_pages_from_file("nonexistent_path_to_file.pdf")
