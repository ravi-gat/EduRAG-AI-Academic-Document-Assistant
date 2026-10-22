"""Answer generation module using Google Gen AI SDK (Gemini) with strict grounding."""

import logging
from typing import List, Optional
from google import genai
from google.genai import types

from src.config import (
    DEFAULT_GEMINI_MODEL,
    SYSTEM_INSTRUCTION,
    INSUFFICIENT_EVIDENCE_RESPONSE,
    DEFAULT_SIMILARITY_THRESHOLD,
    AnswerResult,
    RetrievalResult,
    get_gemini_api_key,
)
from src.citations import build_context_block, validate_and_extract_citations

logger = logging.getLogger(__name__)


def generate_answer(
    question: str,
    retrieved_chunks: List[RetrievalResult],
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_GEMINI_MODEL,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    client: Optional[genai.Client] = None,
) -> AnswerResult:
    """Generate an evidence-grounded answer to a student question.

    Stages:
    1. Input validation.
    2. Evidence sufficiency check (retrieval presence & score threshold).
    3. Grounded prompt construction.
    4. Gemini API invocation.
    5. Citation extraction and validation.

    Args:
        question: Student's natural language question.
        retrieved_chunks: Passages retrieved from FAISS index.
        api_key: Optional Gemini API key. If not provided, loaded from config/env.
        model_name: Configurable Gemini model identifier.
        similarity_threshold: Cosine similarity cutoff for evidence relevance.
        client: Optional pre-configured genai.Client instance (useful for testing/mocking).

    Returns:
        AnswerResult object containing answer text, citations, and evidence flags.
    """
    clean_question = question.strip() if question else ""
    if not clean_question:
        return AnswerResult(
            question="",
            answer="Please enter a valid question.",
            has_sufficient_evidence=False,
            model_used=model_name,
        )

    # Stage 1: Check if any documents or passages were retrieved
    if not retrieved_chunks:
        return AnswerResult(
            question=clean_question,
            answer=INSUFFICIENT_EVIDENCE_RESPONSE,
            has_sufficient_evidence=False,
            retrieved_chunks=[],
            model_used=model_name,
        )

    # Stage 2: Similarity threshold calibration check
    top_score = max(r.similarity_score for r in retrieved_chunks)
    if top_score < similarity_threshold:
        logger.info(
            f"Top similarity score ({top_score:.4f}) below threshold ({similarity_threshold:.4f}). "
            "Returning insufficient evidence fallback."
        )
        return AnswerResult(
            question=clean_question,
            answer=INSUFFICIENT_EVIDENCE_RESPONSE,
            has_sufficient_evidence=False,
            retrieved_chunks=retrieved_chunks,
            model_used=model_name,
        )

    # Stage 3: Retrieve API key
    active_key = api_key or get_gemini_api_key()
    if not active_key and client is None:
        return AnswerResult(
            question=clean_question,
            answer="",
            has_sufficient_evidence=False,
            retrieved_chunks=retrieved_chunks,
            model_used=model_name,
            error_message=(
                "Gemini API key is not configured. Please enter your Gemini API key in the sidebar "
                "or set GEMINI_API_KEY in your .env file."
            ),
        )

    # Stage 4: Construct strictly grounded prompt
    context_text, id_to_result = build_context_block(retrieved_chunks)

    prompt = f"""You are answering questions for a student based strictly on the uploaded course materials below.

CONTEXT PASSAGES:
{context_text}

STUDENT QUESTION:
{clean_question}

INSTRUCTIONS:
1. Answer the question clearly, concisely, and factually, using ONLY information present in the CONTEXT PASSAGES.
2. For each substantive statement or fact, cite the supporting source using bracket tags (e.g., [Source 1], [Source 2]).
3. Use bullet points or numbered lists where helpful for educational clarity.
4. If the CONTEXT PASSAGES do not contain sufficient evidence to answer the question completely and accurately, respond EXACTLY with:
"{INSUFFICIENT_EVIDENCE_RESPONSE}"
Do NOT guess, fabricate, or incorporate external training knowledge.
"""

    # Stage 5: Call Gemini API (with exponential retry on temporary 503 spikes)
    import time
    answer_text = ""
    last_exception = None

    for attempt in range(3):
        try:
            active_client = client or genai.Client(api_key=active_key)
            response = active_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.1,  # Low temperature to prioritize factual fidelity
                ),
            )
            answer_text = response.text.strip() if response and response.text else ""
            last_exception = None
            break
        except Exception as e:
            last_exception = e
            err_str = str(e)
            if ("503" in err_str or "UNAVAILABLE" in err_str) and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            break

    if last_exception is not None:
        err_str = str(last_exception)
        logger.error(f"Gemini API generation error: {err_str}")

        # Provide user-friendly actionable error messages
        if "API_KEY_INVALID" in err_str or "400" in err_str and "key" in err_str.lower():
            friendly_err = "The provided Gemini API key is invalid. Please check your key in the sidebar."
        elif "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
            friendly_err = "Gemini API rate limit or quota exceeded. Please wait a moment before trying again."
        elif "NOT_FOUND" in err_str or "model" in err_str.lower():
            friendly_err = f"The selected model '{model_name}' was not found or is unsupported. Try selecting a different model."
        else:
            friendly_err = f"API error occurred during generation: {err_str}"

        return AnswerResult(
            question=clean_question,
            answer="",
            has_sufficient_evidence=False,
            retrieved_chunks=retrieved_chunks,
            model_used=model_name,
            error_message=friendly_err,
        )

    # Stage 6: Evaluate if the model triggered the insufficient evidence fallback
    normalized_answer = answer_text.strip().lower()
    insufficient_triggers = [
        "i don't know based on the provided documents",
        "cannot be determined from the uploaded documents",
        "not enough information in the provided",
        "does not contain sufficient evidence",
    ]

    if any(trigger in normalized_answer for trigger in insufficient_triggers):
        return AnswerResult(
            question=clean_question,
            answer=INSUFFICIENT_EVIDENCE_RESPONSE,
            has_sufficient_evidence=False,
            retrieved_chunks=retrieved_chunks,
            model_used=model_name,
        )

    # Stage 7: Validate citations against real retrieved chunks
    citations = validate_and_extract_citations(
        llm_answer=answer_text,
        retrieved_chunks=retrieved_chunks,
        min_similarity=similarity_threshold,
    )

    return AnswerResult(
        question=clean_question,
        answer=answer_text,
        citations=citations,
        has_sufficient_evidence=True,
        retrieved_chunks=retrieved_chunks,
        model_used=model_name,
    )
