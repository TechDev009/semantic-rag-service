"""Embedding and generation protocols plus deterministic and OpenAI adapters.

The application depends on the protocols, not on an LLM framework.  The local
implementations are deterministic and intentionally boring, which makes tests
and demos reproducible.  The OpenAI adapters use the current ``openai`` async
client but are only constructed when explicitly configured.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .config import Settings
from .domain import GenerationResult, ProviderHealth, estimate_tokens


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class GenerationProvider(Protocol):
    @property
    def model(self) -> str: ...

    async def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult: ...


_WORD_RE = re.compile(r"[\w][\w'-]*", re.UNICODE)
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def words(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def _feature(token: str, salt: str, dimensions: int) -> tuple[int, float]:
    digest = hashlib.sha256(f"{salt}:{token}".encode()).digest()
    index = int.from_bytes(digest[:8], "big") % dimensions
    sign = 1.0 if digest[8] & 1 else -1.0
    return index, sign


class DeterministicEmbeddingProvider:
    """A stable hashed bag-of-words/character embedding with no network calls."""

    def __init__(self, dimensions: int = 64, model: str = "local-hash-v1") -> None:
        if dimensions < 8:
            raise ValueError("embedding dimensions must be at least 8")
        self._dimensions = dimensions
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _one(self, text: str) -> list[float]:
        vector = np.zeros(self._dimensions, dtype=np.float32)
        tokens = words(text)
        if not tokens:
            return vector.tolist()

        for token in tokens:
            index, sign = _feature(token, "word", self._dimensions)
            vector[index] += sign
            # A second independent projection reduces collisions while keeping
            # the representation compact enough for local demos.
            index, sign = _feature(token, "word-2", self._dimensions)
            vector[index] += 0.5 * sign
            if len(token) >= 3:
                for start in range(len(token) - 2):
                    index, sign = _feature(token[start : start + 3], "tri", self._dimensions)
                    vector[index] += 0.08 * sign

        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.astype(np.float32).tolist()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]


class DeterministicFakeLLM:
    """A safe, deterministic JSON-producing fake for local development.

    The RAG service places retrieved chunks in a JSON data envelope.  This fake
    reads only that envelope and the question; it never executes instructions
    found in chunk text.  Its behavior is intentionally simple, not a quality
    benchmark for a real model.
    """

    def __init__(self, model: str = "fake-rag-v1") -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @staticmethod
    def _extract(prompt: str) -> tuple[str, list[dict[str, str]]]:
        question_match = re.search(r"QUESTION_JSON\s*(.*?)\s*END_QUESTION_JSON", prompt, re.S)
        chunks_match = re.search(
            r"RETRIEVED_CHUNKS_JSON\s*(.*?)\s*END_RETRIEVED_CHUNKS_JSON", prompt, re.S
        )
        if not question_match or not chunks_match:
            return "", []
        try:
            question = str(json.loads(question_match.group(1)))
            raw_chunks = json.loads(chunks_match.group(1))
            chunks = [
                {"id": str(item["id"]), "text": str(item["text"])}
                for item in raw_chunks
                if isinstance(item, dict) and "id" in item and "text" in item
            ]
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            return "", []
        return question, chunks

    @staticmethod
    def _safe_sentence(text: str, query_terms: set[str]) -> str:
        candidates = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        unsafe = ("ignore previous", "ignore all", "system message", "jailbreak", "assistant:")
        for candidate in candidates:
            normalized = set(words(candidate))
            if normalized & query_terms and not any(
                marker in candidate.lower() for marker in unsafe
            ):
                return candidate.strip()[:600]
        for candidate in candidates:
            if candidate.strip() and not any(marker in candidate.lower() for marker in unsafe):
                return candidate.strip()[:600]
        return ""

    async def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        del system_prompt  # The fake follows the protocol, not arbitrary document text.
        question, chunks = self._extract(user_prompt)
        query_terms = {word for word in words(question) if word not in _STOP_WORDS}
        ranked: list[tuple[int, dict[str, str]]] = []
        for chunk in chunks:
            chunk_terms = set(words(chunk["text"]))
            ranked.append((len(query_terms & chunk_terms), chunk))
        ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
        useful = [(score, chunk) for score, chunk in ranked if score > 0]

        if not useful:
            payload = {
                "answer": "I don't have enough evidence in the retrieved documents to answer that.",
                "citation_ids": [],
                "abstained": True,
            }
        else:
            selected = useful[:2]
            snippets = [self._safe_sentence(chunk["text"], query_terms) for _, chunk in selected]
            snippets = [snippet for snippet in snippets if snippet]
            answer = "According to the retrieved documents: " + " ".join(snippets)
            payload = {
                "answer": answer[:20000],
                "citation_ids": [chunk["id"] for _, chunk in selected],
                "abstained": False,
            }
        encoded = json.dumps(payload, sort_keys=True)
        return GenerationResult(
            text=encoded,
            model=self._model,
            prompt_tokens=estimate_tokens(user_prompt),
            completion_tokens=estimate_tokens(encoded),
            estimated_cost_usd=0.0,
        )


class OpenAIEmbeddingProvider:
    """OpenAI embeddings adapter; import and client creation happen on opt-in use."""

    def __init__(
        self, api_key: str, model: str, dimensions: int, base_url: str | None = None
    ) -> None:
        from openai import AsyncOpenAI

        self._model = model
        self._dimensions = dimensions
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=list(texts),
            dimensions=self._dimensions,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]


class OpenAICompatibleLLM:
    """OpenAI Chat Completions adapter with strict JSON schema output."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        from openai import AsyncOpenAI

        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "grounded_rag_answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "answer": {"type": "string"},
                            "citation_ids": {"type": "array", "items": {"type": "string"}},
                            "abstained": {"type": "boolean"},
                        },
                        "required": ["answer", "citation_ids", "abstained"],
                    },
                },
            },
        )
        content = response.choices[0].message.content or ""
        usage: Any = response.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return GenerationResult(
            text=content,
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def build_providers(settings: Settings) -> tuple[EmbeddingProvider, GenerationProvider]:
    """Construct configured providers, keeping local defaults credential-free."""

    if settings.embedding_provider == "local":
        embeddings: EmbeddingProvider = DeterministicEmbeddingProvider(
            dimensions=settings.embedding_dimensions,
            model=settings.embedding_model,
        )
    elif settings.embedding_provider in {"openai", "openai-compatible"}:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI embedding provider")
        embeddings = OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            dimensions=settings.embedding_dimensions,
            base_url=settings.openai_base_url,
        )
    else:
        raise ValueError(f"unsupported embedding provider: {settings.embedding_provider}")

    if settings.generation_provider == "fake":
        generation: GenerationProvider = DeterministicFakeLLM(settings.generation_model)
    elif settings.generation_provider in {"openai", "openai-compatible"}:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI generation provider")
        generation = OpenAICompatibleLLM(
            api_key=settings.openai_api_key,
            model=settings.openai_generation_model,
            base_url=settings.openai_base_url,
        )
    else:
        raise ValueError(f"unsupported generation provider: {settings.generation_provider}")
    return embeddings, generation


def provider_health(
    settings: Settings,
    embedding_provider: EmbeddingProvider,
    generation_provider: GenerationProvider,
) -> list[ProviderHealth]:
    """Return configuration health without making a paid network request."""

    embedding_ready = settings.embedding_provider == "local" or bool(settings.openai_api_key)
    generation_ready = settings.generation_provider == "fake" or bool(settings.openai_api_key)
    return [
        ProviderHealth(
            name=f"embedding:{embedding_provider.model}",
            ready=embedding_ready,
            detail=None if embedding_ready else "provider credentials are missing",
        ),
        ProviderHealth(
            name=f"generation:{generation_provider.model}",
            ready=generation_ready,
            detail=None if generation_ready else "provider credentials are missing",
        ),
    ]
