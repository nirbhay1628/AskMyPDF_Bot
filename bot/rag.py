from __future__ import annotations

import logging
import re
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List

import faiss
import numpy as np
from openai import AsyncOpenAI, OpenAIError

from .utils import chunk_text, extract_text_from_pdf

logger = logging.getLogger(__name__)


class LRUCache:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._data: OrderedDict[str, object] = OrderedDict()

    def get(self, key: str) -> object | None:
        value = self._data.get(key)
        if value is None:
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: object) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self.max_size:
            self._data.popitem(last=False)


@dataclass(slots=True)
class RetrievalCandidate:
    index: int
    score: float
    dense_score: float
    lexical_score: float


@dataclass(slots=True)
class UserRAGSession:
    openai_api_key: str
    embedding_model: str
    chat_model: str
    top_k: int = 4
    retrieval_pool_size: int = 12
    max_context_chars: int = 6000
    min_similarity_score: float = 0.20
    lexical_alpha: float = 0.20
    answer_cache_size: int = 64
    embedding_cache_size: int = 256
    chunk_size_tokens: int = 380
    overlap_tokens: int = 60
    max_history_messages: int = 6

    client: AsyncOpenAI = field(init=False)
    index: faiss.Index | None = field(default=None, init=False)
    dense_matrix: np.ndarray | None = field(default=None, init=False)
    chunks: List[str] = field(default_factory=list, init=False)
    chunk_term_sets: List[set[str]] = field(default_factory=list, init=False)
    source_filename: str | None = field(default=None, init=False)
    history: List[dict] = field(default_factory=list, init=False)
    answer_cache: LRUCache = field(init=False)
    query_embedding_cache: LRUCache = field(init=False)

    _use_local_embeddings: bool = field(default=False, init=False)
    _local_embedding_dim: int = field(default=768, init=False)

    def __post_init__(self) -> None:
        self.client = AsyncOpenAI(api_key=self.openai_api_key)
        self.answer_cache = LRUCache(self.answer_cache_size)
        self.query_embedding_cache = LRUCache(self.embedding_cache_size)

    async def ingest_pdf(self, pdf_path: str, filename: str) -> int:
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(
            text,
            chunk_size_tokens=self.chunk_size_tokens,
            overlap_tokens=self.overlap_tokens,
        )

        if not chunks:
            raise ValueError("Could not create text chunks from PDF.")

        unique_chunks: List[str] = []
        seen_chunks: set[str] = set()
        for chunk in chunks:
            normalized = " ".join(chunk.split())
            if len(normalized) < 40 or normalized in seen_chunks:
                continue
            seen_chunks.add(normalized)
            unique_chunks.append(normalized)

        if not unique_chunks:
            raise ValueError("Could not build usable chunks from the PDF.")

        embeddings = await self._embed_texts(unique_chunks)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        self.index = index
        self.dense_matrix = embeddings
        self.chunks = unique_chunks
        self.chunk_term_sets = [self._tokenize_to_set(chunk) for chunk in unique_chunks]
        self.source_filename = filename
        self.history.clear()
        self.answer_cache = LRUCache(self.answer_cache_size)
        self.query_embedding_cache = LRUCache(self.embedding_cache_size)

        logger.info(
            "Indexed PDF '%s' into %d chunks for current user session.",
            filename,
            len(unique_chunks),
        )
        return len(unique_chunks)

    async def answer_query(self, query: str) -> str:
        if self.index is None or not self.chunks:
            raise RuntimeError("No PDF has been indexed yet for this session.")

        normalized_query = " ".join(query.lower().split())
        if normalized_query:
            cached_answer = self.answer_cache.get(normalized_query)
            if isinstance(cached_answer, str):
                return cached_answer

        candidates = await self.retrieve_relevant_chunks(query, k=self.top_k)
        if not candidates:
            no_context_answer = (
                "I could not find relevant information for this question in the uploaded PDF."
            )
            self._update_history(query, no_context_answer)
            self.answer_cache.set(normalized_query, no_context_answer)
            return no_context_answer

        context_blocks: List[str] = []
        context_chars = 0
        for idx, candidate in enumerate(candidates, start=1):
            chunk = self.chunks[candidate.index]
            block = (
                f"[Chunk {idx}] (dense={candidate.dense_score:.3f}, "
                f"lexical={candidate.lexical_score:.3f}, score={candidate.score:.3f})\n"
                f"{chunk}"
            )
            if context_chars + len(block) > self.max_context_chars:
                break
            context_blocks.append(block)
            context_chars += len(block)

        if not context_blocks:
            no_context_answer = (
                "I could not find enough relevant context in the uploaded PDF to answer that."
            )
            self._update_history(query, no_context_answer)
            self.answer_cache.set(normalized_query, no_context_answer)
            return no_context_answer

        context_text = "\n\n---\n\n".join(context_blocks)

        system_prompt = (
            "You are a precise document QA assistant. "
            "Use only the retrieved context chunks from the uploaded PDF. "
            "Do not use outside knowledge. "
            "If the answer is not present, explicitly say: 'I cannot find this in the uploaded PDF.' "
            "Keep answers concise and factual. "
            "When possible, cite chunk IDs like [Chunk 1], [Chunk 2]."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if self.history:
            messages.extend(self.history[-self.max_history_messages :])
        messages.append(
            {
                "role": "user",
                "content": (
                    f"PDF filename: {self.source_filename or 'Unknown'}\n\n"
                    f"Relevant context:\n{context_text}\n\n"
                    "Instructions:\n"
                    "1) Answer the question using only the context above.\n"
                    "2) If uncertain or missing, state that it is not found in the PDF.\n"
                    "3) Prefer bullet points for multi-part answers.\n\n"
                    f"Question: {query}"
                ),
            }
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.0,
            )
            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                answer = "I could not generate an answer for that question."
        except OpenAIError as exc:
            answer = self._build_api_fallback_answer(query, exc, context_blocks)

        self._update_history(query, answer)
        self.answer_cache.set(normalized_query, answer)
        return answer

    async def retrieve_relevant_chunks(self, query: str, k: int = 4) -> List[RetrievalCandidate]:
        if self.index is None or not self.chunks:
            return []

        query_vector = await self._embed_query(query)
        faiss.normalize_L2(query_vector)

        pool_size = min(max(k, self.retrieval_pool_size), len(self.chunks))
        scores, indices = self.index.search(query_vector, pool_size)

        query_terms = self._tokenize_to_set(query)
        candidates: List[RetrievalCandidate] = []
        for dense_score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx == -1:
                continue
            lexical_score = self._lexical_overlap(query_terms, self.chunk_term_sets[idx])
            fused_score = ((1.0 - self.lexical_alpha) * dense_score) + (
                self.lexical_alpha * lexical_score
            )
            candidates.append(
                RetrievalCandidate(
                    index=idx,
                    score=fused_score,
                    dense_score=dense_score,
                    lexical_score=lexical_score,
                )
            )

        filtered = [c for c in candidates if c.score >= self.min_similarity_score]
        if not filtered:
            return []

        filtered.sort(key=lambda c: c.score, reverse=True)
        return self._mmr_select(filtered, query_vector[0], top_k=min(k, len(filtered)))

    async def _embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        vectors: List[List[float]] = []

        if self._use_local_embeddings:
            vectors = self._local_embed_texts(texts)
        else:
            try:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    response = await self.client.embeddings.create(
                        model=self.embedding_model,
                        input=batch,
                    )
                    vectors.extend(item.embedding for item in response.data)
            except OpenAIError as exc:
                logger.warning(
                    "OpenAI embeddings unavailable; using local hashed embeddings fallback: %s",
                    exc,
                )
                self._use_local_embeddings = True
                vectors = self._local_embed_texts(texts)

        matrix = np.array(vectors, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("Failed to generate embeddings for document chunks.")
        return matrix

    async def _embed_query(self, query: str) -> np.ndarray:
        cache_key = " ".join(query.lower().split())
        cached = self.query_embedding_cache.get(cache_key)
        if isinstance(cached, np.ndarray):
            return cached.copy()

        if self._use_local_embeddings:
            vector = np.array([self._local_embed_texts([query])[0]], dtype="float32")
        else:
            try:
                response = await self.client.embeddings.create(
                    model=self.embedding_model,
                    input=query,
                )
                vector = np.array([response.data[0].embedding], dtype="float32")
            except OpenAIError as exc:
                logger.warning(
                    "OpenAI query embedding unavailable; using local hashed fallback: %s",
                    exc,
                )
                self._use_local_embeddings = True
                vector = np.array([self._local_embed_texts([query])[0]], dtype="float32")

        self.query_embedding_cache.set(cache_key, vector.copy())
        return vector

    def _build_api_fallback_answer(
        self,
        query: str,
        exc: OpenAIError,
        context_blocks: List[str],
    ) -> str:
        fallback_summary = self._build_extractive_summary(query, context_blocks)

        prefix = "OpenAI API is currently unavailable for generation"
        if "quota" in str(exc).lower() or "429" in str(exc):
            prefix = "OpenAI API quota/rate limit reached"

        if fallback_summary:
            return (
                f"{prefix}, so I cannot generate a full model answer right now.\n\n"
                f"Based on the retrieved PDF sections, {fallback_summary}"
            )

        return (
            f"{prefix}, and I cannot generate an answer right now. "
            "Please retry shortly or check your API key/billing."
        )

    def _build_extractive_summary(self, query: str, context_blocks: List[str]) -> str:
        if not context_blocks:
            return ""

        context_text = "\n".join(self._strip_chunk_header(block) for block in context_blocks)
        sentences = self._split_sentences(context_text)
        if not sentences:
            return ""

        combined_text = context_text.lower()
        topic = self._infer_topic(combined_text)

        if self._is_definition_query(query):
            definition = self._build_definition_style_answer(query, sentences, topic)
            if definition:
                return definition

        scored_sentences = []
        for sentence in sentences:
            score = self._score_sentence(sentence, topic)
            if score > 0:
                scored_sentences.append((score, sentence))

        if not scored_sentences:
            scored_sentences = [(1.0, s) for s in sentences[:3]]

        scored_sentences.sort(key=lambda item: item[0], reverse=True)
        selected: List[str] = []
        seen: set[str] = set()
        for _, sentence in scored_sentences:
            normalized = sentence.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            selected.append(normalized)
            if len(selected) >= 3:
                break

        if not selected:
            return ""

        lead = ""
        if topic:
            lead = f"the document sections suggest that {topic} is discussed in terms of "
        else:
            lead = "the document sections suggest that "

        joined = "; ".join(selected)
        if not joined.endswith("."):
            joined += "."
        return f"{lead}{joined}"

    def _is_definition_query(self, query: str) -> bool:
        normalized = query.strip().lower()
        return bool(
            re.match(r"^(what is|what's|what are|define|define\s+the|explain)\b", normalized)
        )

    def _extract_query_subject(self, query: str) -> str:
        cleaned = query.strip().rstrip("?.,!")
        match = re.match(
            r"^(?:what is|what's|what are|define|explain)\s+(?:the\s+)?(.+)$",
            cleaned,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        return cleaned

    def _build_definition_style_answer(
        self,
        query: str,
        sentences: List[str],
        topic: str,
    ) -> str:
        subject = self._extract_query_subject(query)
        subject_terms = self._tokenize_to_set(subject)
        if not subject_terms:
            return ""

        scored: List[tuple[float, str]] = []
        for sentence in sentences:
            sentence_terms = self._tokenize_to_set(sentence)
            overlap = len(subject_terms.intersection(sentence_terms))
            score = overlap * 4.0

            if overlap > 0:
                score += 2.0
            if any(marker in sentence.lower() for marker in ["is", "means", "refers", "defines", "describes"]):
                score += 2.0
            if len(sentence) < 220:
                score += 0.5
            if topic and any(term in sentence.lower() for term in topic.split()):
                score += 1.0

            if score > 0:
                scored.append((score, sentence))

        if not scored:
            return ""

        scored.sort(key=lambda item: item[0], reverse=True)
        best_sentences = [sentence for _, sentence in scored[:2]]
        first_sentence = best_sentences[0]

        if subject.lower() not in first_sentence.lower():
            lead = f"{subject} refers to "
        else:
            lead = "It refers to "

        explanation = first_sentence
        if len(best_sentences) > 1 and best_sentences[1] != first_sentence:
            explanation = f"{first_sentence} Also, {best_sentences[1].rstrip('.')}"

        explanation = explanation.strip()
        if explanation.endswith(":"):
            explanation = explanation[:-1].strip()

        if explanation.lower().startswith(subject.lower()):
            return explanation if explanation.endswith(".") else f"{explanation}."

        if not lead.endswith(" "):
            lead += " "

        result = f"{subject} is a concept discussed in the PDF. {lead}{explanation}"
        result = re.sub(r"\s+", " ", result).strip()
        if not result.endswith("."):
            result += "."
        return result

    def _strip_chunk_header(self, block: str) -> str:
        parts = block.split("\n", 1)
        return parts[1].strip() if len(parts) > 1 else block.strip()

    def _split_sentences(self, text: str) -> List[str]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
        sentences: List[str] = []
        for sentence in raw_sentences:
            stripped = sentence.strip(" -•\t\r\n")
            if len(stripped) >= 20:
                sentences.append(stripped)
        return sentences

    def _infer_topic(self, text: str) -> str:
        keyword_groups = [
            ("linear programming problem", ["lpp", "linear programming", "linear programming problem"]),
            ("primal and dual formulations", ["primal", "dual", "duality", "complementary slackness"]),
            ("binding constraints and shadow prices", ["binding", "shadow price", "slackness", "constraint"]),
            ("transport or optimization modeling", ["transport", "optimization", "modeling", "pulp"]),
        ]

        for label, keywords in keyword_groups:
            if any(keyword in text for keyword in keywords):
                return label

        return "the main topic"

    def _score_sentence(self, sentence: str, topic: str) -> float:
        sentence_lower = sentence.lower()
        score = 0.0

        topic_keywords = set(re.findall(r"[a-z0-9]{2,}", topic.lower()))
        sentence_terms = self._tokenize_to_set(sentence_lower)

        overlap = len(topic_keywords.intersection(sentence_terms))
        score += overlap * 2.0

        query_like_terms = {term for term in topic_keywords if term not in {"the", "and", "main", "topic"}}
        score += len(query_like_terms.intersection(sentence_terms))

        if any(marker in sentence_lower for marker in ["is", "means", "refers", "includes", "covers", "discusses"]):
            score += 1.0
        if any(marker in sentence_lower for marker in ["problem", "solution", "constraint", "objective", "dual", "primal"]):
            score += 1.0
        if len(sentence) < 200:
            score += 0.5

        return score

    def _update_history(self, question: str, answer: str) -> None:
        if self.max_history_messages <= 0:
            return
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        if len(self.history) > self.max_history_messages:
            self.history = self.history[-self.max_history_messages :]

    def _mmr_select(
        self,
        candidates: List[RetrievalCandidate],
        query_vector: np.ndarray,
        top_k: int,
        lambda_param: float = 0.75,
    ) -> List[RetrievalCandidate]:
        if self.dense_matrix is None:
            return candidates[:top_k]

        selected: List[RetrievalCandidate] = []
        remaining = candidates.copy()

        while remaining and len(selected) < top_k:
            if not selected:
                best = max(remaining, key=lambda c: c.score)
                selected.append(best)
                remaining.remove(best)
                continue

            best_candidate: RetrievalCandidate | None = None
            best_mmr = -1e9

            for candidate in remaining:
                doc_vec = self.dense_matrix[candidate.index]
                relevance = float(np.dot(query_vector, doc_vec))

                max_similarity_to_selected = max(
                    float(np.dot(doc_vec, self.dense_matrix[s.index])) for s in selected
                )

                mmr_score = (lambda_param * relevance) - (
                    (1.0 - lambda_param) * max_similarity_to_selected
                )
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_candidate = candidate

            if best_candidate is None:
                break

            selected.append(best_candidate)
            remaining.remove(best_candidate)

        return selected

    def _tokenize_to_set(self, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]{2,}", text.lower()))

    def _lexical_overlap(self, query_terms: set[str], chunk_terms: set[str]) -> float:
        if not query_terms or not chunk_terms:
            return 0.0
        intersection = len(query_terms.intersection(chunk_terms))
        return intersection / max(1, len(query_terms))

    def _local_embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            vec = np.zeros(self._local_embedding_dim, dtype="float32")
            for token in re.findall(r"[a-z0-9]{2,}", text.lower()):
                idx = zlib.crc32(token.encode("utf-8")) % self._local_embedding_dim
                vec[idx] += 1.0
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            vectors.append(vec.tolist())
        return vectors
