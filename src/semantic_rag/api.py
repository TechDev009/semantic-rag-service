"""FastAPI application and versioned HTTP contracts."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import Settings, get_settings
from .db import ConflictError, Database, NotFoundError
from .domain import AnswerRecord, CollectionRecord, FeedbackRecord, SearchHit
from .providers import EmbeddingProvider, GenerationProvider, build_providers, provider_health
from .rag import ProviderOutputError, RagService


class CollectionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be blank")
        return value


class DocumentIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=500)
    source: str | None = Field(default=None, max_length=2000)
    text: str | None = Field(default=None, max_length=2_000_000)
    chunks: list[str] | None = Field(default=None, max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_content(self) -> DocumentIngestRequest:
        if self.text is None and self.chunks is None:
            raise ValueError("provide text or chunks")
        if self.text is not None and self.chunks is not None:
            raise ValueError("provide text or chunks, not both")
        if self.chunks is not None and not any(chunk.strip() for chunk in self.chunks):
            raise ValueError("chunks must contain non-whitespace content")
        if self.text is not None and not self.text.strip():
            raise ValueError("text must contain non-whitespace content")
        return self

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("title must not be blank")
        return value


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection_id: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class AnswerRequest(BaseModel):
    """Accept the search-style ``query`` and the human-friendly ``question`` spelling."""

    model_config = ConfigDict(extra="forbid")

    collection_id: str = Field(min_length=1, max_length=100)
    query: str | None = Field(default=None, min_length=1, max_length=4000)
    question: str | None = Field(default=None, min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def validate_question(self) -> AnswerRequest:
        if (self.query is None) == (self.question is None):
            raise ValueError("provide exactly one of query or question")
        value = (self.query or self.question or "").strip()
        if not value:
            raise ValueError("question must not be blank")
        self.query = value
        return self

    @property
    def prompt(self) -> str:
        assert self.query is not None
        return self.query


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: Literal["positive", "negative"]
    comment: str | None = Field(default=None, max_length=4000)


class CollectionResponse(CollectionRecord):
    pass


class DocumentResponse(BaseModel):
    document_id: str
    collection_id: str
    title: str
    source: str | None
    chunk_ids: list[str]
    chunk_count: int
    embedding_model: str
    embedding_dimensions: int


class SearchHitResponse(SearchHit):
    pass


class SearchResponse(BaseModel):
    collection_id: str
    query: str
    hits: list[SearchHitResponse]


class CitationResponse(BaseModel):
    chunk_id: str
    document_id: str
    quote: str
    score: float
    rank: int


class UsageResponse(BaseModel):
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    embedding_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class AnswerResponse(BaseModel):
    id: str
    collection_id: str
    question: str
    answer: str
    abstained: bool
    citations: list[CitationResponse]
    usage: UsageResponse
    created_at: Any


class FeedbackResponse(FeedbackRecord):
    pass


@dataclass
class AppState:
    settings: Settings
    database: Database
    embeddings: EmbeddingProvider
    generation: GenerationProvider
    rag: RagService


def _answer_response(answer: AnswerRecord) -> AnswerResponse:
    return AnswerResponse.model_validate(answer.model_dump())


def _feedback_response(feedback: FeedbackRecord) -> FeedbackResponse:
    return FeedbackResponse.model_validate(feedback.model_dump())


def create_app(
    settings: Settings | None = None,
    database: Database | None = None,
    embeddings: EmbeddingProvider | None = None,
    generation: GenerationProvider | None = None,
) -> FastAPI:
    runtime_settings = settings or get_settings()
    if embeddings is None or generation is None:
        configured_embeddings, configured_generation = build_providers(runtime_settings)
        embeddings = embeddings or configured_embeddings
        generation = generation or configured_generation
    runtime_database = database or Database(runtime_settings)
    rag = RagService(runtime_database, embeddings, generation, runtime_settings)
    state = AppState(runtime_settings, runtime_database, embeddings, generation, rag)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await runtime_database.initialize()
        app.state.runtime = state
        yield
        await runtime_database.dispose()

    app = FastAPI(
        title="Semantic RAG Service",
        version="0.1.0",
        description="Provider-agnostic semantic search and grounded RAG API",
        lifespan=lifespan,
    )
    app.state.runtime = state

    @app.exception_handler(NotFoundError)
    async def not_found_handler(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

    @app.exception_handler(ConflictError)
    async def conflict_handler(_: Request, exc: ConflictError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": str(exc)})

    @app.exception_handler(ProviderOutputError)
    async def provider_handler(_: Request, exc: ProviderOutputError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content={"detail": str(exc)})

    @app.get("/v1/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/health/ready")
    async def ready(request: Request) -> JSONResponse:
        current: AppState = request.app.state.runtime
        database_ready = await current.database.ping()
        providers = provider_health(current.settings, current.embeddings, current.generation)
        ready_now = database_ready and all(provider.ready for provider in providers)
        payload = {
            "status": "ready" if ready_now else "not_ready",
            "database": database_ready,
            "providers": [provider.model_dump() for provider in providers],
        }
        return JSONResponse(
            status_code=status.HTTP_200_OK if ready_now else status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload,
        )

    @app.post(
        "/v1/collections",
        response_model=CollectionResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_collection(
        request: CollectionCreateRequest, http_request: Request
    ) -> CollectionRecord:
        current: AppState = http_request.app.state.runtime
        return await current.database.create_collection(
            request.name,
            request.description,
            current.embeddings.model,
            current.embeddings.dimensions,
        )

    @app.post(
        "/v1/documents",
        response_model=DocumentResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def ingest_document(
        request: DocumentIngestRequest, http_request: Request
    ) -> DocumentResponse:
        current: AppState = http_request.app.state.runtime
        try:
            document, chunk_ids = await current.rag.ingest_document(
                collection_id=request.collection_id,
                title=request.title,
                source=request.source,
                metadata=request.metadata,
                text=request.text,
                chunks=request.chunks,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return DocumentResponse(
            document_id=document.id,
            collection_id=document.collection_id,
            title=document.title,
            source=document.source,
            chunk_ids=chunk_ids,
            chunk_count=len(chunk_ids),
            embedding_model=current.embeddings.model,
            embedding_dimensions=current.embeddings.dimensions,
        )

    @app.post("/v1/search", response_model=SearchResponse)
    async def search(request: SearchRequest, http_request: Request) -> SearchResponse:
        current: AppState = http_request.app.state.runtime
        await current.database.get_collection(request.collection_id)
        query_vector = (await current.embeddings.embed([request.query]))[0]
        hits = await current.database.search(
            request.collection_id, query_vector, request.query, request.top_k
        )
        return SearchResponse(
            collection_id=request.collection_id,
            query=request.query,
            hits=[SearchHitResponse.model_validate(hit.model_dump()) for hit in hits],
        )

    @app.post("/v1/answers", response_model=AnswerResponse)
    async def answer(request: AnswerRequest, http_request: Request) -> AnswerResponse:
        current: AppState = http_request.app.state.runtime
        result = await current.rag.answer(request.collection_id, request.prompt, request.top_k)
        return _answer_response(result)

    @app.post(
        "/v1/answers/{answer_id}/feedback",
        response_model=FeedbackResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def feedback(
        answer_id: str, request: FeedbackRequest, http_request: Request
    ) -> FeedbackResponse:
        current: AppState = http_request.app.state.runtime
        result = await current.database.create_feedback(answer_id, request.rating, request.comment)
        return _feedback_response(result)

    return app
