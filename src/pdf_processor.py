"""PDF processing and text extraction module using PyMuPDF (fitz)."""

import io
import re
import logging
from typing import List, Dict, Any, Union, Optional
import pymupdf as fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Base exception for PDF extraction issues."""
    pass


class EmptyPDFError(PDFExtractionError):
    """Raised when PDF contains zero pages or is zero bytes."""
    pass


class ScannedPDFError(PDFExtractionError):
    """Raised when PDF has pages but contains no extractable digital text (OCR required)."""
    pass


def normalize_text(text: str) -> str:
    """Normalize extracted text whitespace while preserving paragraph boundaries.

    - Replaces non-breaking spaces with standard spaces.
    - Strips isolated control characters.
    - Preserves meaningful newlines while condensing multiple blank lines into a single blank line.
    - Removes hyphenation line breaks (e.g., 'trans-\nformer' -> 'transformer').
    """
    if not text:
        return ""

    # Replace non-breaking spaces and form feeds
    text = text.replace("\xa0", " ").replace("\x0c", "\n")

    # Fix line-broken hyphenations (e.g. 'struc-\nture' -> 'structure')
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Condense consecutive horizontal spaces/tabs
    text = re.sub(r'[ \t]+', ' ', text)

    # Condense 3+ newlines into 2 (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def extract_pages_from_bytes(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extract text from PDF raw bytes page-by-page.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file.
        filename: Original filename of the document.

    Returns:
        List of dictionaries with keys:
            - source: str (filename)
            - page: int (1-based PDF page number)
            - text: str (normalized extracted text)
            - word_count: int

    Raises:
        EmptyPDFError: If file is empty or has 0 pages.
        PDFExtractionError: If PDF is corrupted or unreadable.
        ScannedPDFError: If PDF has pages but zero extractable text across all pages.
    """
    if not file_bytes or len(file_bytes) == 0:
        raise EmptyPDFError(f"Uploaded file '{filename}' is empty (0 bytes).")

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        logger.error(f"Failed to open PDF '{filename}': {e}")
        raise PDFExtractionError(
            f"Could not read PDF '{filename}'. File may be corrupted, password-protected, or invalid format: {e}"
        ) from e

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        raise EmptyPDFError(f"PDF document '{filename}' contains 0 pages.")

    pages_data = []
    total_words = 0

    try:
        for page_idx in range(total_pages):
            page = doc[page_idx]
            raw_text = page.get_text("text") or ""
            clean_text = normalize_text(raw_text)
            word_count = len(clean_text.split()) if clean_text else 0
            total_words += word_count

            # Store every page so page numbering strictly matches 1-based PDF index
            pages_data.append({
                "source": filename,
                "page": page_idx + 1,  # 1-based page index
                "text": clean_text,
                "word_count": word_count,
            })
    finally:
        doc.close()

    # Detect if the PDF is scanned or image-only
    if total_words == 0:
        raise ScannedPDFError(
            f"'{filename}' ({total_pages} pages) contains no selectable or extractable digital text. "
            "This document appears to be a scanned image or photo PDF. Optical Character Recognition (OCR) "
            "is required to extract text from scanned documents."
        )

    return pages_data


def extract_pages_from_file(file_path: str, filename: Optional[str] = None) -> List[Dict[str, Any]]:
    """Extract text from a PDF file on disk.

    Args:
        file_path: Path to the local PDF file.
        filename: Optional override for the filename recorded in metadata.
    """
    import os
    if not os.path.exists(file_path):
        raise PDFExtractionError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    doc_name = filename or os.path.basename(file_path)
    return extract_pages_from_bytes(file_bytes, doc_name)
