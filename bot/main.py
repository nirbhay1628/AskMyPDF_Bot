import logging

from dotenv import load_dotenv
from telegram.error import Conflict, NetworkError, TimedOut
from telegram.ext import Application

from .handlers import register_handlers
from .utils import load_settings


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    load_dotenv(override=True)
    configure_logging()

    settings = load_settings()

    app = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .connect_timeout(30.0)
        .read_timeout(30.0)
        .write_timeout(30.0)
        .pool_timeout(30.0)
        .build()
    )

    register_handlers(
        app,
        config={
            "openai_api_key": settings.openai_api_key,
            "openai_chat_model": settings.openai_chat_model,
            "openai_embedding_model": settings.openai_embedding_model,
            "top_k": settings.top_k,
            "retrieval_pool_size": settings.retrieval_pool_size,
            "max_context_chars": settings.max_context_chars,
            "min_similarity_score": settings.min_similarity_score,
            "lexical_alpha": settings.lexical_alpha,
            "answer_cache_size": settings.answer_cache_size,
            "embedding_cache_size": settings.embedding_cache_size,
            "max_history_messages": settings.max_history_messages,
        },
    )

    try:
        app.run_polling(drop_pending_updates=True, bootstrap_retries=3)
    except Conflict:
        logging.getLogger(__name__).error(
            "Bot startup failed due to Telegram polling conflict. "
            "Stop other running instances that use the same bot token and start again."
        )
        raise
    except TimedOut:
        logging.getLogger(__name__).error(
            "Bot startup timed out while reaching Telegram API. "
            "Check internet/VPN/firewall and retry."
        )
        raise
    except NetworkError as exc:
        logging.getLogger(__name__).error(
            "Bot startup failed due to Telegram network error: %s", exc
        )
        raise


if __name__ == "__main__":
    main()
