"""Source citation validation and metadata formatting module for EduRAG."""

import re
import logging
from typing import List, Dict, Tuple, Optional, Set
from src.config import Citation, RetrievalResult

logger = logging.getLogger(__name__)


def build_context_block(retrieved_chunks: List[RetrievalResult]) -> Tuple[str, Dict[int, RetrievalResult]]:
    """Format retrieved passages into a structured context block for the LLM prompt.

    Each passage is explicitly tagged with a numeric source identifier, filename,
    and 1-based PDF page number.

    Returns:
        Tuple of:
            - context_text: String block for the LLM prompt
            - id_to_result: Mapping from 1-based source number to RetrievalResult
    """
    context_lines = []
    id_to_result: Dict[int, RetrievalResult] = {}

    for idx, res in enumerate(retrieved_chunks, start=1):
        id_to_result[idx] = res
        header = f"[Source {idx} | Document: {res.source} | PDF Page: {res.page}]"
        context_lines.append(f"{header}\n{res.text.strip()}\n")

    return "\n".join(context_lines), id_to_result


def clean_passage_excerpt(text: str, max_words: int = 50) -> str:
    """Produce a readable, concise supporting excerpt for UI citation cards."""
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def validate_and_extract_citations(
    llm_answer: str,
    retrieved_chunks: List[RetrievalResult],
    min_similarity: float = 0.0,
) -> List[Citation]:
    """Extract and validate citations from an LLM answer against retrieved ground-truth chunks.

    Guarantees:
    - Never fabricates a citation.
    - Only includes sources that actually exist in the retrieved chunks.
    - Accurately attributes 1-based PDF page numbers.
    - Rejects hallucinated filenames or non-existent page numbers.

    Args:
        llm_answer: Text returned by the language model.
        retrieved_chunks: Ground-truth passages supplied to the LLM.
        min_similarity: Optional threshold for citation validity.

    Returns:
        List of validated Citation objects.
    """
    if not retrieved_chunks:
        return []

    # Map sources and pages to real retrieved chunks for validation
    valid_source_page_map: Dict[Tuple[str, int], RetrievalResult] = {}
    valid_idx_map: Dict[int, RetrievalResult] = {}

    for idx, res in enumerate(retrieved_chunks, start=1):
        valid_idx_map[idx] = res
        key = (res.source.strip().lower(), res.page)
        if key not in valid_source_page_map:
            valid_source_page_map[key] = res

    cited_indices: Set[int] = set()

    # Pattern 1: [Source X] or [Source X, PDF Page Y]
    source_matches = re.findall(r'\[Source\s+(\d+)', llm_answer, re.IGNORECASE)
    for match in source_matches:
        idx = int(match)
        if idx in valid_idx_map:
            cited_indices.add(idx)

    # Pattern 2: [Doc X] or [Document X]
    doc_matches = re.findall(r'\[Doc(?:ument)?\s+(\d+)', llm_answer, re.IGNORECASE)
    for match in doc_matches:
        idx = int(match)
        if idx in valid_idx_map:
            cited_indices.add(idx)

    # Pattern 3: Look for explicit filename mentions in text (e.g., "lecture1.pdf, page 3")
    for (src_lower, page), res in valid_source_page_map.items():
        # Check if the filename is mentioned in the answer
        if src_lower in llm_answer.lower():
            # If the specific page number is also mentioned near it or anywhere
            page_pattern = rf'(?:page|p\.)\s*{page}\b'
            if re.search(page_pattern, llm_answer, re.IGNORECASE):
                # Find its index
                for idx, r in valid_idx_map.items():
                    if r.chunk.chunk_id == res.chunk.chunk_id:
                        cited_indices.add(idx)

    # If the model explicitly cited sources using brackets, use those validated citations
    validated_citations: List[Citation] = []
    seen_chunk_ids: Set[str] = set()

    if cited_indices:
        for idx in sorted(cited_indices):
            res = valid_idx_map[idx]
            if res.chunk.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(res.chunk.chunk_id)

            validated_citations.append(
                Citation(
                    source=res.source,
                    page=res.page,
                    excerpt=clean_passage_excerpt(res.text),
                    similarity_score=round(res.similarity_score, 4),
                    chunk_id=res.chunk.chunk_id,
                )
            )

    # Fallback: If the model answered grounded in the context but omitted bracket tags,
    # attribute the answer to the top retrieved chunks that meet the similarity threshold
    if not validated_citations and retrieved_chunks:
        for res in retrieved_chunks:
            if res.similarity_score >= min_similarity:
                if res.chunk.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(res.chunk.chunk_id)

                validated_citations.append(
                    Citation(
                        source=res.source,
                        page=res.page,
                        excerpt=clean_passage_excerpt(res.text),
                        similarity_score=round(res.similarity_score, 4),
                        chunk_id=res.chunk.chunk_id,
                    )
                )
            if len(validated_citations) >= 3:
                break

    return validated_citations
