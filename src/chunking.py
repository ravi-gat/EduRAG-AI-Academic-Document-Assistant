"""Document chunking module for academic text.

Splits extracted pages into overlapping chunks while preserving source metadata:
- chunk_id: Stable unique identifier for the chunk
- source: Original filename
- page: 1-based PDF page number
- text: The passage text
- word_count: Number of words in passage
"""

import re
from typing import List, Dict, Any, Optional
from src.config import (
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DocumentChunk,
)


def split_text_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> List[str]:
    """Split text into overlapping chunks using word boundaries.

    Args:
        text: Plain text content to chunk.
        chunk_size: Maximum number of words per chunk (default 200).
        chunk_overlap: Number of overlapping words between consecutive chunks (default 40).

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    total_words = len(words)

    # Edge case: text fits into a single chunk
    if total_words <= chunk_size:
        return [" ".join(words)]

    # Guard against invalid overlap
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    step = max(1, chunk_size - chunk_overlap)
    chunks = []

    start_idx = 0
    while start_idx < total_words:
        end_idx = min(start_idx + chunk_size, total_words)
        chunk_words = words[start_idx:end_idx]
        chunk_str = " ".join(chunk_words)

        if chunk_str.strip():
            chunks.append(chunk_str)

        if end_idx >= total_words:
            break

        start_idx += step

    return chunks


def generate_chunk_id(source: str, page: int, chunk_idx: int) -> str:
    """Generate a clean, deterministic chunk ID.

    Format: <sanitized_source>_p<page>_c<chunk_idx>
    """
    clean_name = re.sub(r'[^a-zA-Z0-9_\-.]', '_', source)
    return f"{clean_name}_p{page}_c{chunk_idx}"


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> List[DocumentChunk]:
    """Chunk a list of extracted page dictionaries while preserving page attribution.

    Args:
        pages: List of dicts with 'source', 'page', 'text'.
        chunk_size: Target words per chunk.
        chunk_overlap: Overlapping words between consecutive chunks.

    Returns:
        List of DocumentChunk instances.
    """
    all_chunks: List[DocumentChunk] = []

    for page_info in pages:
        source = page_info.get("source", "unknown.pdf")
        page_num = page_info.get("page", 1)
        raw_text = page_info.get("text", "")

        if not raw_text or not raw_text.strip():
            continue

        passages = split_text_into_chunks(
            text=raw_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for idx, passage in enumerate(passages):
            chunk_id = generate_chunk_id(source, page_num, idx)
            word_count = len(passage.split())
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                source=source,
                page=page_num,
                text=passage,
                word_count=word_count,
            )
            all_chunks.append(chunk)

    return all_chunks
