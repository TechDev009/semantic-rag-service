"""Provider-independent domain types shared by storage, RAG, and the API."""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CollectionRecord(BaseModel):
    id: str
    name: str
    description: str | None = None
    embedding_model: str
    embedding_dimensions: int
    created_at: datetime


class DocumentRecord(BaseModel):
    id: str
    collection_id: str
    title: str
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ChunkRecord(BaseModel):
    id: str
    collection_id: str
    document_id: str
    chunk_index: int
    text: str
    embedding_model: str
    embedding_dimensions: int
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    """A chunk returned by hybrid vector/keyword retrieval."""

    chunk_id: str
    document_id: str
    collection_id: str
    chunk_index: int
    text: str
    score: float
    vector_score: float
    keyword_score: float
    title: str
    source: str | None = None


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    quote: str
    score: float
    rank: int = Field(ge=1)


class Usage(BaseModel):
    """Usage is persisted even for local providers to make cost accounting explicit."""

    provider: str
    model: str
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    embedding_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)

    def with_total(self) -> "Usage":
        return self.model_copy(
            update={
                "total_tokens": self.prompt_tokens + self.completion_tokens + self.embedding_tokens
            }
        )


class AnswerRecord(BaseModel):
    id: str
    collection_id: str
    question: str
    answer: str
    abstained: bool
    citations: list[Citation] = Field(default_factory=list)
    usage: Usage
    created_at: datetime


class FeedbackRecord(BaseModel):
    id: str
    answer_id: str
    rating: Literal["positive", "negative"]
    comment: str | None = None
    created_at: datetime


class GeneratedAnswer(BaseModel):
    """The only output shape accepted from a generation provider."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=20000)
    citation_ids: list[str] = Field(default_factory=list, max_length=20)
    abstained: bool = False

    @field_validator("citation_ids")
    @classmethod
    def unique_citations(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class GenerationResult(BaseModel):
    text: str
    model: str
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)


class ProviderHealth(BaseModel):
    name: str
    ready: bool
    detail: str | None = None


def utc_now() -> datetime:
    return datetime.now(UTC)


def estimate_tokens(text: str) -> int:
    """A deterministic conservative estimate for providers without tokenizer access."""

    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)
