import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz

logger = logging.getLogger(__name__)

MAX_PDF_SIZE_MB = 15
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024


@dataclass(slots=True)
class Settings:
    telegram_bot_token: str
    openai_api_key: str
    openai_chat_model: str
    openai_embedding_model: str
    top_k: int
    retrieval_pool_size: int
    max_context_chars: int
    min_similarity_score: float
    lexical_alpha: float
    answer_cache_size: int
    embedding_cache_size: int
    max_history_messages: int


def load_settings() -> Settings:
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    openai_api_key = (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("GEMINI_API_KEY", "").strip()
    )
    openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini").strip()
    openai_embedding_model = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    ).strip()

    top_k_raw = os.getenv("RAG_TOP_K", "4").strip()
    retrieval_pool_raw = os.getenv("RAG_RETRIEVAL_POOL_SIZE", "12").strip()
    max_context_chars_raw = os.getenv("RAG_MAX_CONTEXT_CHARS", "6000").strip()
    min_similarity_raw = os.getenv("RAG_MIN_SIMILARITY", "0.20").strip()
    lexical_alpha_raw = os.getenv("RAG_LEXICAL_ALPHA", "0.20").strip()
    answer_cache_size_raw = os.getenv("RAG_ANSWER_CACHE_SIZE", "64").strip()
    embedding_cache_size_raw = os.getenv("RAG_EMBEDDING_CACHE_SIZE", "256").strip()
    max_history_raw = os.getenv("RAG_MAX_HISTORY_MESSAGES", "6").strip()

    try:
        top_k = max(1, int(top_k_raw))
    except ValueError:
        logger.warning("Invalid RAG_TOP_K value '%s'. Falling back to 4.", top_k_raw)
        top_k = 4

    try:
        retrieval_pool_size = max(top_k, int(retrieval_pool_raw))
    except ValueError:
        logger.warning(
            "Invalid RAG_RETRIEVAL_POOL_SIZE value '%s'. Falling back to 12.",
            retrieval_pool_raw,
        )
        retrieval_pool_size = max(top_k, 12)

    try:
        max_context_chars = max(1000, int(max_context_chars_raw))
    except ValueError:
        logger.warning(
            "Invalid RAG_MAX_CONTEXT_CHARS value '%s'. Falling back to 6000.",
            max_context_chars_raw,
        )
        max_context_chars = 6000

    try:
        min_similarity_score = float(min_similarity_raw)
    except ValueError:
        logger.warning(
            "Invalid RAG_MIN_SIMILARITY value '%s'. Falling back to 0.20.",
            min_similarity_raw,
        )
        min_similarity_score = 0.20

    try:
        lexical_alpha = float(lexical_alpha_raw)
    except ValueError:
        logger.warning(
            "Invalid RAG_LEXICAL_ALPHA value '%s'. Falling back to 0.20.",
            lexical_alpha_raw,
        )
        lexical_alpha = 0.20
    lexical_alpha = min(max(0.0, lexical_alpha), 1.0)

    try:
        answer_cache_size = max(1, int(answer_cache_size_raw))
    except ValueError:
        logger.warning(
            "Invalid RAG_ANSWER_CACHE_SIZE value '%s'. Falling back to 64.",
            answer_cache_size_raw,
        )
        answer_cache_size = 64

    try:
        embedding_cache_size = max(1, int(embedding_cache_size_raw))
    except ValueError:
        logger.warning(
            "Invalid RAG_EMBEDDING_CACHE_SIZE value '%s'. Falling back to 256.",
            embedding_cache_size_raw,
        )
        embedding_cache_size = 256

    try:
        max_history_messages = max(0, int(max_history_raw))
    except ValueError:
        logger.warning(
            "Invalid RAG_MAX_HISTORY_MESSAGES value '%s'. Falling back to 6.",
            max_history_raw,
        )
        max_history_messages = 6

    if not telegram_bot_token:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN in environment.")
    if not openai_api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment.")

    return Settings(
        telegram_bot_token=telegram_bot_token,
        openai_api_key=openai_api_key,
        openai_chat_model=openai_chat_model,
        openai_embedding_model=openai_embedding_model,
        top_k=top_k,
        retrieval_pool_size=retrieval_pool_size,
        max_context_chars=max_context_chars,
        min_similarity_score=min_similarity_score,
        lexical_alpha=lexical_alpha,
        answer_cache_size=answer_cache_size,
        embedding_cache_size=embedding_cache_size,
        max_history_messages=max_history_messages,
    )


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    text_chunks: List[str] = []
    with fitz.open(path) as doc:
        if doc.page_count == 0:
            raise ValueError("The PDF appears to be empty.")

        for page_index, page in enumerate(doc, start=1):
            page_text = page.get_text("text").strip()
            if page_text:
                text_chunks.append(page_text)
            else:
                logger.debug("Page %d had no extractable text.", page_index)

    full_text = "\n\n".join(text_chunks).strip()
    if not full_text:
        raise ValueError(
            "No extractable text found in the PDF. It may be scanned/image-based."
        )

    return full_text


def chunk_text(text: str, chunk_size_tokens: int = 380, overlap_tokens: int = 60) -> List[str]:
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be greater than 0.")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative.")
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_size_tokens.")

    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    step = chunk_size_tokens - overlap_tokens

    for start in range(0, len(words), step):
        end = start + chunk_size_tokens
        chunk_words = words[start:end]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break

    return chunks


def safe_delete_file(path: str | Path) -> None:
    file_path = Path(path)
    if file_path.exists():
        try:
            file_path.unlink()
        except OSError as exc:
            logger.warning("Could not delete temp file '%s': %s", file_path, exc)
