import hashlib
import math
import re
from typing import Protocol, runtime_checkable

from openai import AsyncOpenAI

from app.config import EMBEDDING_DIM, settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for embedding generation and RAG answer synthesis."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate 1536-dimensional embedding vectors for a batch of texts."""
        ...

    async def generate_answer(self, query: str, context_chunks: list[str]) -> str:
        """Synthesize an answer to `query` grounded in `context_chunks`."""
        ...


class DeterministicFakeProvider:
    """Offline, deterministic LLM & Embedding provider for fast (<2s) tests.

    Uses signed feature hashing over lowercased alphanumeric tokens plus a small
    whole-text digest component to produce unit-normalized 1536-d vectors.
    Texts sharing vocabulary have higher cosine similarity than unrelated texts,
    and identical strings produce identical vectors.
    """

    def __init__(self, dimension: int = EMBEDDING_DIM) -> None:
        self.dimension = dimension
        self.embed_calls: list[list[str]] = []
        self.answer_calls: list[tuple[str, list[str]]] = []

    def _embed_single(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        tokens = _TOKEN_RE.findall(text.lower())

        if tokens:
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                # Project each token into 8 coordinates with deterministic signs
                for i in range(0, 32, 4):
                    raw = int.from_bytes(digest[i : i + 4], "big")
                    idx = raw % self.dimension
                    sign = 1.0 if (raw & 1) == 0 else -1.0
                    vec[idx] += sign

        # Add a small whole-string signature so distinct strings with identical token
        # bags or empty/symbol-only strings still get a valid deterministic direction.
        full_digest = hashlib.sha256(text.encode("utf-8")).digest()
        for i in range(0, 32, 4):
            raw = int.from_bytes(full_digest[i : i + 4], "big")
            idx = raw % self.dimension
            sign = 1.0 if (raw & 1) == 0 else -1.0
            vec[idx] += 0.15 * sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [self._embed_single(t) for t in texts]

    async def generate_answer(self, query: str, context_chunks: list[str]) -> str:
        self.answer_calls.append((query, list(context_chunks)))
        if not context_chunks:
            return f"No relevant context found for query: {query}"
        top_snippet = context_chunks[0].strip()
        return f"Based on the provided documents ({len(context_chunks)} chunks): {top_snippet} [Query: {query}]"


class OpenAIProvider:
    """Production OpenAI provider implementing LLMProvider."""

    def __init__(
        self,
        api_key: str | None = None,
        embedding_model: str = settings.embedding_model,
        chat_model: str = settings.chat_model,
    ) -> None:
        self.client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)
        self.embedding_model = embedding_model
        self.chat_model = chat_model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def generate_answer(self, query: str, context_chunks: list[str]) -> str:
        context_block = "\n\n---\n\n".join(context_chunks)
        response = await self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise technical assistant. Answer the user's question "
                        "strictly using the provided document context."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_block}\n\nQuestion: {query}",
                },
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""
