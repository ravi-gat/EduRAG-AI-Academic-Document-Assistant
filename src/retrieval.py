"""FAISS vector retrieval and index management module for EduRAG."""

import logging
from typing import List, Dict, Optional, Set
import numpy as np
import faiss

from src.config import (
    EMBEDDING_DIMENSION,
    DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    DocumentChunk,
    RetrievalResult,
)
from src.embeddings import generate_embeddings

logger = logging.getLogger(__name__)


class FAISSIndexManager:
    """Manages the FAISS IndexFlatIP vector index and corresponding chunk metadata."""

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.chunks: List[DocumentChunk] = []
        # Track indexed documents and their chunk IDs
        self.doc_to_chunk_indices: Dict[str, List[int]] = {}

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def total_documents(self) -> int:
        return len(self.doc_to_chunk_indices)

    @property
    def document_names(self) -> List[str]:
        return list(self.doc_to_chunk_indices.keys())

    def get_document_stats(self) -> Dict[str, Dict[str, int]]:
        """Return statistics for each indexed document (pages, chunks, words)."""
        stats: Dict[str, Dict[str, int]] = {}
        for doc_name, indices in self.doc_to_chunk_indices.items():
            doc_chunks = [self.chunks[i] for i in indices]
            unique_pages = len(set(c.page for c in doc_chunks))
            total_words = sum(c.word_count for c in doc_chunks)
            stats[doc_name] = {
                "pages": unique_pages,
                "chunks": len(doc_chunks),
                "words": total_words,
            }
        return stats

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> int:
        """Add chunks and their normalized embeddings to the FAISS index.

        Args:
            chunks: List of DocumentChunk instances.
            embeddings: Float32 array of shape (N, dimension).

        Returns:
            Number of chunks added.
        """
        if len(chunks) == 0:
            return 0

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Chunk count ({len(chunks)}) does not match embeddings shape ({embeddings.shape[0]})"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) does not match index dimension ({self.dimension})"
            )

        # Ensure float32 format
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        start_idx = len(self.chunks)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

        # Update document to chunk mapping
        for offset, chunk in enumerate(chunks):
            idx = start_idx + offset
            if chunk.source not in self.doc_to_chunk_indices:
                self.doc_to_chunk_indices[chunk.source] = []
            self.doc_to_chunk_indices[chunk.source].append(idx)

        logger.info(
            f"Added {len(chunks)} chunks to FAISS index. Total index size: {self.total_chunks}"
        )
        return len(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = 0.0,
        deduplicate: bool = True,
    ) -> List[RetrievalResult]:
        """Search the FAISS index for top-k chunks most similar to the query.

        Args:
            query_embedding: Float32 query vector of shape (1, dimension).
            top_k: Number of passages to retrieve.
            threshold: Minimum cosine similarity score to include.
            deduplicate: Whether to collapse identical or heavily overlapping passages.

        Returns:
            List of RetrievalResult instances sorted by similarity descending.
        """
        if self.total_chunks == 0 or query_embedding.shape[0] == 0:
            return []

        # Ensure float32
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        # FAISS search
        effective_k = min(top_k * 2 if deduplicate else top_k, self.total_chunks)
        if effective_k <= 0:
            return []

        scores, indices = self.index.search(query_embedding, effective_k)

        results: List[RetrievalResult] = []
        seen_texts: Set[str] = set()

        for score, idx in zip(scores[0], indices[0]):
            # FAISS returns -1 for empty slots
            if idx < 0 or idx >= len(self.chunks):
                continue

            float_score = float(score)

            # Filter below threshold if requested
            if float_score < threshold:
                continue

            chunk = self.chunks[idx]

            # Simple deduplication: avoid showing nearly identical passages
            normalized_snippet = " ".join(chunk.text.split()[:30]).lower()
            if deduplicate and normalized_snippet in seen_texts:
                continue
            seen_texts.add(normalized_snippet)

            results.append(RetrievalResult(chunk=chunk, similarity_score=float_score))

            if len(results) >= top_k:
                break

        return results

    def remove_document(
        self,
        source_filename: str,
        remaining_embeddings: Optional[np.ndarray] = None,
    ) -> bool:
        """Remove all chunks associated with a specific document and rebuild the index.

        This guarantees that no stale chunks or phantom indices remain in FAISS.

        Args:
            source_filename: Filename to remove.
            remaining_embeddings: Optional precomputed embeddings for the remaining chunks.

        Returns:
            True if document was found and removed, False otherwise.
        """
        if source_filename not in self.doc_to_chunk_indices:
            logger.warning(f"Document '{source_filename}' not found in active index.")
            return False

        # Filter remaining chunks
        remaining_chunks = [c for c in self.chunks if c.source != source_filename]

        # Reset state
        self.clear()

        if remaining_chunks:
            if remaining_embeddings is not None:
                embs = remaining_embeddings
            else:
                texts = [c.text for c in remaining_chunks]
                embs = generate_embeddings(texts)
            self.add_chunks(remaining_chunks, embs)

        logger.info(
            f"Removed document '{source_filename}'. New index chunk count: {self.total_chunks}"
        )
        return True

    def clear(self) -> None:
        """Clear all chunks and reset the FAISS index."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        self.doc_to_chunk_indices = {}
        logger.info("Cleared FAISS index and chunk metadata.")
