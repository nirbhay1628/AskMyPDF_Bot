from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from telegram import Update
from telegram.error import Conflict, NetworkError, TimedOut
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .rag import UserRAGSession
from .utils import MAX_PDF_SIZE_BYTES, safe_delete_file

logger = logging.getLogger(__name__)

USER_SESSIONS: dict[int, UserRAGSession] = {}


def register_handlers(app: Application, config: dict) -> None:
    app.bot_data["config"] = config

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf_upload))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_question)
    )
    app.add_error_handler(global_error_handler)


async def global_error_handler(
    update: object, context: ContextTypes.DEFAULT_TYPE
) -> None:
    error = context.error
    if isinstance(error, Conflict):
        logger.error(
            "Telegram polling conflict detected. Ensure only one bot instance is running."
        )
        return
    if isinstance(error, TimedOut):
        logger.warning("Telegram request timed out. The bot will keep retrying.")
        return
    if isinstance(error, NetworkError):
        logger.warning("Telegram network error: %s", error)
        return
    logger.exception("Unhandled exception in update handler", exc_info=error)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send me a PDF file, and then ask questions about it.\n\n"
        "Commands:\n"
        "/start - Show instructions\n"
        "/help - Show help"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "How to use this bot:\n"
        "1) Upload a PDF document.\n"
        "2) Wait until processing is complete.\n"
        "3) Ask questions in plain text.\n\n"
        "Each user has an independent in-memory FAISS index for their uploaded PDF."
    )


async def handle_pdf_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return

    doc = update.message.document
    if doc is None:
        await update.message.reply_text("Please upload a valid PDF document.")
        return

    if doc.file_size and doc.file_size > MAX_PDF_SIZE_BYTES:
        max_mb = MAX_PDF_SIZE_BYTES // (1024 * 1024)
        await update.message.reply_text(
            f"The file is too large. Maximum supported size is {max_mb} MB."
        )
        return

    config = context.application.bot_data.get("config", {})
    processing_msg = await update.message.reply_text(
        "Processing your PDF and building embeddings. Please wait..."
    )

    temp_pdf_path: str | None = None
    try:
        telegram_file = await doc.get_file()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_pdf_path = temp_file.name

        await telegram_file.download_to_drive(custom_path=temp_pdf_path)

        session = UserRAGSession(
            openai_api_key=config["openai_api_key"],
            embedding_model=config["openai_embedding_model"],
            chat_model=config["openai_chat_model"],
            top_k=config["top_k"],
            retrieval_pool_size=config["retrieval_pool_size"],
            max_context_chars=config["max_context_chars"],
            min_similarity_score=config["min_similarity_score"],
            lexical_alpha=config["lexical_alpha"],
            answer_cache_size=config["answer_cache_size"],
            embedding_cache_size=config["embedding_cache_size"],
            max_history_messages=config["max_history_messages"],
        )
        chunks_count = await session.ingest_pdf(
            temp_pdf_path,
            filename=doc.file_name or Path(temp_pdf_path).name,
        )

        USER_SESSIONS[update.effective_user.id] = session

        await processing_msg.edit_text(
            "PDF processed successfully.\n"
            f"Indexed {chunks_count} chunks.\n"
            "You can now ask questions about the document."
        )
    except Exception as exc:
        logger.exception("OpenAI API or processing error while processing PDF")
        await processing_msg.edit_text(
            f"Failed to process PDF (OpenAI/API error): {exc}"
        )
    finally:
        if temp_pdf_path:
            safe_delete_file(temp_pdf_path)


async def handle_user_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return

    question = (update.message.text or "").strip()
    if not question:
        await update.message.reply_text("Please send a non-empty question.")
        return

    session = USER_SESSIONS.get(update.effective_user.id)
    if session is None:
        await update.message.reply_text(
            "Please upload a PDF first so I can build a searchable index."
        )
        return

    thinking_msg = await update.message.reply_text("Searching the PDF and generating answer...")
    try:
        answer = await session.answer_query(question)
        await thinking_msg.edit_text(answer)
    except Exception as exc:
        logger.exception("Failed to answer user question")
        await thinking_msg.edit_text(f"Failed to answer question (OpenAI/API error): {exc}")
