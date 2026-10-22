"""Tests for answer generation, evidence grounding, and API robustness."""

from unittest.mock import MagicMock
import pytest

from src.config import (
    DocumentChunk,
    RetrievalResult,
    INSUFFICIENT_EVIDENCE_RESPONSE,
)
from src.answer_generator import generate_answer


@pytest.fixture
def mock_retrieved_chunks():
    """Create sample high-similarity chunks."""
    c1 = DocumentChunk(
        chunk_id="cs.pdf_p4_c0",
        source="cs.pdf",
        page=4,
        text="The RSA algorithm was invented in 1977 by Rivest, Shamir, and Adleman.",
        word_count=13,
    )
    return [RetrievalResult(chunk=c1, similarity_score=0.85)]


def test_empty_question():
    """Empty question input should return a prompt to enter a valid question."""
    res = generate_answer(question="", retrieved_chunks=[])
    assert res.has_sufficient_evidence is False
    assert "Please enter a valid question" in res.answer


def test_empty_retrieved_passages():
    """When no passages are retrieved, return insufficient evidence fallback."""
    res = generate_answer(
        question="What is RSA?",
        retrieved_chunks=[],
        api_key="fake-key",
    )
    assert res.has_sufficient_evidence is False
    assert res.answer == INSUFFICIENT_EVIDENCE_RESPONSE
    assert len(res.citations) == 0


def test_low_similarity_below_threshold():
    """When retrieved chunks have similarity lower than threshold, fallback is returned."""
    chunk = DocumentChunk(
        chunk_id="doc.pdf_p1_c0",
        source="doc.pdf",
        page=1,
        text="Unrelated text about culinary recipes.",
        word_count=5,
    )
    low_sim_result = [RetrievalResult(chunk=chunk, similarity_score=0.15)]

    res = generate_answer(
        question="Explain quantum computing.",
        retrieved_chunks=low_sim_result,
        similarity_threshold=0.35,
        api_key="fake-key",
    )
    assert res.has_sufficient_evidence is False
    assert res.answer == INSUFFICIENT_EVIDENCE_RESPONSE


def test_missing_api_key_error(mock_retrieved_chunks, monkeypatch):
    """Missing API key should return an actionable error message rather than crash."""
    monkeypatch.setattr("src.answer_generator.get_gemini_api_key", lambda: None)
    res = generate_answer(
        question="Who invented RSA?",
        retrieved_chunks=mock_retrieved_chunks,
        api_key="",  # No key
        client=None,
    )
    assert res.error_message is not None
    assert "Gemini API key is not configured" in res.error_message


def test_mocked_gemini_successful_answer(mock_retrieved_chunks):
    """Test successful generation using a mocked Gemini client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "RSA was invented in 1977 by Rivest, Shamir, and Adleman [Source 1]."
    mock_client.models.generate_content.return_value = mock_response

    res = generate_answer(
        question="Who invented RSA?",
        retrieved_chunks=mock_retrieved_chunks,
        client=mock_client,
    )

    assert res.has_sufficient_evidence is True
    assert "Rivest, Shamir, and Adleman" in res.answer
    assert len(res.citations) == 1
    assert res.citations[0].source == "cs.pdf"
    assert res.citations[0].page == 4


def test_mocked_gemini_insufficient_evidence(mock_retrieved_chunks):
    """When the model determines context is insufficient, returns exact fallback."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = (
        "I don't know based on the provided documents. "
        "Please upload a relevant document or ask a question supported by your current documents."
    )
    mock_client.models.generate_content.return_value = mock_response

    res = generate_answer(
        question="What is the speed of light?",
        retrieved_chunks=mock_retrieved_chunks,
        client=mock_client,
    )

    assert res.has_sufficient_evidence is False
    assert res.answer == INSUFFICIENT_EVIDENCE_RESPONSE
    assert len(res.citations) == 0


def test_mocked_gemini_api_exception(mock_retrieved_chunks):
    """API exceptions should be caught and returned as readable error messages."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded")

    res = generate_answer(
        question="Who invented RSA?",
        retrieved_chunks=mock_retrieved_chunks,
        client=mock_client,
    )

    assert res.has_sufficient_evidence is False
    assert res.error_message is not None
    assert "rate limit or quota exceeded" in res.error_message.lower()
