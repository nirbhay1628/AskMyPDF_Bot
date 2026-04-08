# Telegram PDF RAG Bot (Python)

A complete async Telegram bot that lets users upload a PDF and chat with it using a manual RAG pipeline:
- PDF text extraction with PyMuPDF
- Chunking into ~300-500 token-sized pieces
- Embeddings via OpenAI API
- FAISS vector search per user session
- Answer generation via OpenAI chat completions

## Project Structure

```text
bot/
  main.py
  handlers.py
  rag.py
  utils.py
requirements.txt
.env.example
README.md
```

## Prerequisites

- Python 3.10+
- A Telegram bot token from BotFather
- OpenAI API key

## Setup

1. Create and activate a virtual environment.

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Create `.env` from `.env.example` and fill in your keys.

```powershell
Copy-Item .env.example .env
```

Then edit `.env`:

```env
TELEGRAM_BOT_TOKEN=...
OPENAI_API_KEY=...
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
RAG_TOP_K=4
RAG_RETRIEVAL_POOL_SIZE=12
RAG_MAX_CONTEXT_CHARS=6000
RAG_MIN_SIMILARITY=0.20
RAG_LEXICAL_ALPHA=0.20
RAG_ANSWER_CACHE_SIZE=64
RAG_EMBEDDING_CACHE_SIZE=256
RAG_MAX_HISTORY_MESSAGES=6
```

## Production Optimizations Included

- Caching:
  - LRU cache for query embeddings
  - LRU cache for repeated question answers
- API cost control:
  - Deduplicates and filters short chunks before embedding
  - Context character budget to cap prompt size
  - Similarity threshold to skip low-confidence chat calls
  - Deterministic answering with `temperature=0.0`
- Retrieval quality:
  - Hybrid scoring: semantic similarity + lexical overlap
  - MMR re-ranking for diversity among top context chunks
- Prompt engineering:
  - Grounded instructions to use only PDF context
  - Explicit fallback when answer is missing in source
  - Chunk-aware citation style (`[Chunk N]`)

## Run

From the project root:

```powershell
python -m bot.main
```

## Usage

1. Start a chat with your bot and send `/start`.
2. Upload a PDF document.
3. Wait for the bot to finish processing (embedding + FAISS indexing).
4. Ask questions in text.

## Notes

- File size limit is set to 15 MB for safer processing.
- Each Telegram user has their own in-memory FAISS index and chat history.
- If the bot restarts, in-memory sessions are cleared.
- Scanned image-only PDFs may fail text extraction unless OCR text is embedded.
