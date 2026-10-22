"""Tests for FAISS vector retrieval and index management."""

import pytest
import numpy as np

from src.config import DocumentChunk, EMBEDDING_DIMENSION
from src.retrieval import FAISSIndexManager


@pytest.fixture
def sample_chunks():
    """Create three distinct sample chunks."""
    c1 = DocumentChunk(
        chunk_id="doc1_p1_c0",
        source="doc1.pdf",
        page=1,
        text="The RSA algorithm is a public key cryptosystem based on prime factorization.",
        word_count=13,
    )
    c2 = DocumentChunk(
        chunk_id="doc1_p2_c0",
        source="doc1.pdf",
        page=2,
        text="MergeSort is a divide-and-conquer sorting algorithm with O(n log n) complexity.",
        word_count=11,
    )
    c3 = DocumentChunk(
        chunk_id="doc2_p1_c0",
        source="doc2.pdf",
        page=1,
        text="Quantum entanglement occurs when pairs of particles interact in correlated ways.",
        word_count=11,
    )
    return [c1, c2, c3]


@pytest.fixture
def synthetic_embeddings():
    """Create synthetic normalized orthogonal vectors for exact similarity testing."""
    # 3 vectors of dimension EMBEDDING_DIMENSION
    vecs = np.zeros((3, EMBEDDING_DIMENSION), dtype=np.float32)
    # Unit vectors in orthogonal directions
    vecs[0, 0] = 1.0  # Vector 0 (RSA)
    vecs[1, 1] = 1.0  # Vector 1 (MergeSort)
    vecs[2, 2] = 1.0  # Vector 2 (Entanglement)
    return vecs


def test_empty_index():
    """Test searching an empty FAISS index returns an empty list safely."""
    manager = FAISSIndexManager(dimension=EMBEDDING_DIMENSION)
    query_vec = np.zeros((1, EMBEDDING_DIMENSION), dtype=np.float32)
    query_vec[0, 0] = 1.0
    results = manager.search(query_vec, top_k=5)
    assert results == []
    assert manager.total_chunks == 0
    assert manager.total_documents == 0


def test_add_chunks_and_search(sample_chunks, synthetic_embeddings):
    """Test indexing chunks and verifying exact top-1 retrieval matching."""
    manager = FAISSIndexManager(dimension=EMBEDDING_DIMENSION)
    added = manager.add_chunks(sample_chunks, synthetic_embeddings)
    assert added == 3
    assert manager.total_chunks == 3
    assert manager.total_documents == 2

    # Query matching Vector 0 (RSA)
    query = np.zeros((1, EMBEDDING_DIMENSION), dtype=np.float32)
    query[0, 0] = 1.0

    results = manager.search(query, top_k=2)
    assert len(results) == 2
    # Top result should be RSA chunk with similarity 1.0
    assert results[0].chunk.chunk_id == "doc1_p1_c0"
    assert results[0].source == "doc1.pdf"
    assert results[0].page == 1
    assert pytest.approx(results[0].similarity_score, 0.01) == 1.0

    # Second result should have 0.0 similarity due to orthogonality
    assert pytest.approx(results[1].similarity_score, 0.01) == 0.0


def test_top_k_limit(sample_chunks, synthetic_embeddings):
    """Test that top_k strictly limits the number of returned results."""
    manager = FAISSIndexManager(dimension=EMBEDDING_DIMENSION)
    manager.add_chunks(sample_chunks, synthetic_embeddings)

    query = np.ones((1, EMBEDDING_DIMENSION), dtype=np.float32) / np.sqrt(EMBEDDING_DIMENSION)
    results = manager.search(query, top_k=1)
    assert len(results) == 1


def test_threshold_filtering(sample_chunks, synthetic_embeddings):
    """Test that threshold excludes low similarity passages."""
    manager = FAISSIndexManager(dimension=EMBEDDING_DIMENSION)
    manager.add_chunks(sample_chunks, synthetic_embeddings)

    query = np.zeros((1, EMBEDDING_DIMENSION), dtype=np.float32)
    query[0, 0] = 1.0  # matches only vector 0

    # With threshold 0.5, only vector 0 (score 1.0) should be returned
    results = manager.search(query, top_k=5, threshold=0.5)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "doc1_p1_c0"


def test_remove_document(sample_chunks, synthetic_embeddings):
    """Test removing a document purges its chunks and rebuilds the index."""
    manager = FAISSIndexManager(dimension=EMBEDDING_DIMENSION)
    manager.add_chunks(sample_chunks, synthetic_embeddings)
    assert "doc1.pdf" in manager.document_names
    assert "doc2.pdf" in manager.document_names

    # Remove doc1.pdf (only chunk 2 from doc2 remains)
    removed = manager.remove_document("doc1.pdf", remaining_embeddings=synthetic_embeddings[2:])
    assert removed is True
    assert "doc1.pdf" not in manager.document_names
    assert "doc2.pdf" in manager.document_names
    assert manager.total_documents == 1
    # Only doc2 chunk remains
    assert manager.total_chunks == 1
    assert manager.chunks[0].source == "doc2.pdf"


def test_document_stats(sample_chunks, synthetic_embeddings):
    """Test retrieval of per-document metadata statistics."""
    manager = FAISSIndexManager(dimension=EMBEDDING_DIMENSION)
    manager.add_chunks(sample_chunks, synthetic_embeddings)

    stats = manager.get_document_stats()
    assert "doc1.pdf" in stats
    assert stats["doc1.pdf"]["pages"] == 2
    assert stats["doc1.pdf"]["chunks"] == 2
    assert "doc2.pdf" in stats
    assert stats["doc2.pdf"]["pages"] == 1
    assert stats["doc2.pdf"]["chunks"] == 1
