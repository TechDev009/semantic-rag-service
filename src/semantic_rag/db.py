"""Async SQLAlchemy storage with a SQLite local path and pgvector production path."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    bindparam,
    select,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import TypeDecorator

from .config import Settings
from .domain import (
    AnswerRecord,
    Citation,
    CollectionRecord,
    DocumentRecord,
    FeedbackRecord,
    SearchHit,
    Usage,
    utc_now,
)

try:
    from pgvector.sqlalchemy import Vector

    PGVECTOR_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency is present in production installs
    Vector = None  # type: ignore[assignment]
    PGVECTOR_AVAILABLE = False


EMBEDDING_VECTOR_DIMENSIONS = 64
_WORD_RE = re.compile(r"[\w][\w'-]*", re.UNICODE)


class EmbeddingType(TypeDecorator[list[float]]):
    """Use JSON locally and a native pgvector column on PostgreSQL.

    The default local model is 64-dimensional.  Production deployments should
    keep that dimension (or create a migration with their chosen dimension).
    """

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql" and PGVECTOR_AVAILABLE and Vector is not None:
            return dialect.type_descriptor(Vector(EMBEDDING_VECTOR_DIMENSIONS))
        return dialect.type_descriptor(JSON())


class Base(DeclarativeBase):
    pass


class CollectionRow(Base):
    __tablename__ = "collections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_model: Mapped[str] = mapped_column(String(200), nullable=False)
    embedding_dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class DocumentRow(Base):
    __tablename__ = "documents"
    __table_args__ = (Index("ix_documents_collection_id", "collection_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    collection_id: Mapped[str] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    source: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    document_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ChunkRow(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
        Index("ix_chunks_collection_id", "collection_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    collection_id: Mapped[str] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), nullable=False
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(EmbeddingType(), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(200), nullable=False)
    embedding_version: Mapped[str] = mapped_column(String(100), nullable=False, default="v1")
    embedding_dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False)


class AnswerRow(Base):
    __tablename__ = "answers"
    __table_args__ = (Index("ix_answers_collection_id", "collection_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    collection_id: Mapped[str] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), nullable=False
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    abstained: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class CitationRow(Base):
    __tablename__ = "citations"
    __table_args__ = (Index("ix_citations_answer_id", "answer_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    answer_id: Mapped[str] = mapped_column(
        ForeignKey("answers.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("chunks.id", ondelete="RESTRICT"), nullable=False
    )
    document_id: Mapped[str] = mapped_column(String(36), nullable=False)
    quote: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)


class UsageRow(Base):
    __tablename__ = "answer_usage"

    answer_id: Mapped[str] = mapped_column(
        ForeignKey("answers.id", ondelete="CASCADE"), primary_key=True
    )
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, nullable=False)


class FeedbackRow(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    answer_id: Mapped[str] = mapped_column(
        ForeignKey("answers.id", ondelete="CASCADE"), nullable=False
    )
    rating: Mapped[str] = mapped_column(String(20), nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RepositoryError(RuntimeError):
    """Base class for expected storage errors."""


class NotFoundError(RepositoryError):
    pass


class ConflictError(RepositoryError):
    pass


def _tokens(text_value: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(text_value)}


def _collection_record(row: CollectionRow) -> CollectionRecord:
    return CollectionRecord(
        id=row.id,
        name=row.name,
        description=row.description,
        embedding_model=row.embedding_model,
        embedding_dimensions=row.embedding_dimensions,
        created_at=row.created_at,
    )


def _document_record(row: DocumentRow) -> DocumentRecord:
    return DocumentRecord(
        id=row.id,
        collection_id=row.collection_id,
        title=row.title,
        source=row.source,
        metadata=row.document_metadata,
        created_at=row.created_at,
    )


class Database:
    """Repository facade used by the API and RAG service.

    SQLite performs an in-process hybrid search over JSON embeddings.  When the
    URL is PostgreSQL and pgvector is installed, initialization creates the
    extension and an HNSW cosine index; retrieval asks PostgreSQL for vector
    candidates before applying the same keyword reranking.
    """

    def __init__(self, settings_or_url: Settings | str) -> None:
        if isinstance(settings_or_url, Settings):
            self.settings = settings_or_url
            url = settings_or_url.async_database_url
        else:
            self.settings = None
            url = settings_or_url
        self.url = url
        kwargs: dict[str, Any] = {"future": True}
        if url.startswith("sqlite") and ":memory:" in url:
            kwargs["connect_args"] = {"check_same_thread": False}
            kwargs["poolclass"] = StaticPool
        self.engine: AsyncEngine = create_async_engine(url, **kwargs)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    @property
    def is_postgres(self) -> bool:
        return self.url.startswith("postgresql+")

    async def initialize(self) -> None:
        async with self.engine.begin() as connection:
            if self.is_postgres:
                if not PGVECTOR_AVAILABLE:
                    raise RuntimeError("pgvector package is required for PostgreSQL")
                await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await connection.run_sync(Base.metadata.create_all)
            if self.is_postgres and PGVECTOR_AVAILABLE:
                await connection.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw "
                        "ON chunks USING hnsw (embedding vector_cosine_ops)"
                    )
                )

    async def dispose(self) -> None:
        await self.engine.dispose()

    async def ping(self) -> bool:
        try:
            async with self.sessions() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    async def create_collection(
        self, name: str, description: str | None, embedding_model: str, embedding_dimensions: int
    ) -> CollectionRecord:
        row = CollectionRow(
            id=str(uuid4()),
            name=name,
            description=description,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            created_at=utc_now(),
        )
        async with self.sessions() as session:
            session.add(row)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise ConflictError("a collection with that name already exists") from exc
        return _collection_record(row)

    async def get_collection(self, collection_id: str) -> CollectionRecord:
        async with self.sessions() as session:
            row = await session.get(CollectionRow, collection_id)
        if row is None:
            raise NotFoundError("collection not found")
        return _collection_record(row)

    async def create_document(
        self,
        collection_id: str,
        title: str,
        source: str | None,
        metadata: dict[str, Any],
        chunks: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        embedding_model: str,
        embedding_dimensions: int,
        embedding_version: str = "v1",
    ) -> tuple[DocumentRecord, list[str]]:
        if len(chunks) != len(embeddings) or not chunks:
            raise ValueError("each document must contain at least one chunk and embedding")
        document_id = str(uuid4())
        document = DocumentRow(
            id=document_id,
            collection_id=collection_id,
            title=title,
            source=source,
            document_metadata=metadata,
            created_at=utc_now(),
        )
        chunk_ids: list[str] = []
        async with self.sessions() as session:
            if await session.get(CollectionRow, collection_id) is None:
                raise NotFoundError("collection not found")
            session.add(document)
            for index, (chunk_text, vector) in enumerate(zip(chunks, embeddings, strict=True)):
                chunk_id = str(uuid4())
                chunk_ids.append(chunk_id)
                session.add(
                    ChunkRow(
                        id=chunk_id,
                        collection_id=collection_id,
                        document_id=document_id,
                        chunk_index=index,
                        text=chunk_text,
                        embedding=[float(value) for value in vector],
                        embedding_model=embedding_model,
                        embedding_version=embedding_version,
                        embedding_dimensions=embedding_dimensions,
                        token_count=max(1, len(_tokens(chunk_text))),
                        chunk_metadata={},
                    )
                )
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise ConflictError("document could not be stored") from exc
        return _document_record(document), chunk_ids

    async def search(
        self, collection_id: str, query_embedding: Sequence[float], query: str, top_k: int
    ) -> list[SearchHit]:
        await self.get_collection(collection_id)
        query_vector = np.asarray(query_embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vector))
        query_terms = _tokens(query)
        async with self.sessions() as session:
            if self.is_postgres and PGVECTOR_AVAILABLE and Vector is not None:
                distance = ChunkRow.embedding.op("<=>")(
                    bindparam(
                        "query_embedding",
                        value=list(map(float, query_embedding)),
                        type_=Vector(len(query_embedding)),
                    )
                )
                statement = (
                    select(ChunkRow, DocumentRow, distance.label("distance"))
                    .join(DocumentRow, DocumentRow.id == ChunkRow.document_id)
                    .where(ChunkRow.collection_id == collection_id)
                    .order_by(distance)
                    .limit(max(top_k * 4, top_k))
                )
                result = await session.execute(statement)
                database_rows = result.all()
                candidates = [
                    (chunk, document, max(0.0, min(1.0, 1.0 - float(distance_value))))
                    for chunk, document, distance_value in database_rows
                ]
            else:
                statement = (
                    select(ChunkRow, DocumentRow)
                    .join(DocumentRow, DocumentRow.id == ChunkRow.document_id)
                    .where(ChunkRow.collection_id == collection_id)
                )
                result = await session.execute(statement)
                candidates = [(chunk, document, None) for chunk, document in result.all()]

        scored: list[SearchHit] = []
        for chunk, document, database_vector_score in candidates:
            stored = np.asarray(chunk.embedding, dtype=np.float32)
            if database_vector_score is None:
                stored_norm = float(np.linalg.norm(stored))
                vector_score = (
                    float(np.dot(query_vector, stored) / (query_norm * stored_norm))
                    if query_norm and stored_norm
                    else 0.0
                )
            else:
                vector_score = database_vector_score
            overlap = len(query_terms & _tokens(chunk.text))
            keyword_score = overlap / len(query_terms) if query_terms else 0.0
            combined = max(-1.0, min(1.0, 0.78 * vector_score + 0.22 * keyword_score))
            scored.append(
                SearchHit(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    collection_id=chunk.collection_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=combined,
                    vector_score=vector_score,
                    keyword_score=keyword_score,
                    title=document.title,
                    source=document.source,
                )
            )
        scored.sort(key=lambda item: (-item.score, item.chunk_id))
        return scored[:top_k]

    async def persist_answer(
        self,
        collection_id: str,
        question: str,
        answer: str,
        abstained: bool,
        citations: Sequence[Citation],
        usage: Usage,
    ) -> AnswerRecord:
        answer_id = str(uuid4())
        created_at = utc_now()
        answer_row = AnswerRow(
            id=answer_id,
            collection_id=collection_id,
            question=question,
            answer=answer,
            abstained=abstained,
            created_at=created_at,
        )
        async with self.sessions() as session:
            if await session.get(CollectionRow, collection_id) is None:
                raise NotFoundError("collection not found")
            session.add(answer_row)
            for citation in citations:
                session.add(
                    CitationRow(
                        id=str(uuid4()),
                        answer_id=answer_id,
                        chunk_id=citation.chunk_id,
                        document_id=citation.document_id,
                        quote=citation.quote,
                        score=citation.score,
                        rank=citation.rank,
                    )
                )
            session.add(
                UsageRow(
                    answer_id=answer_id,
                    provider=usage.provider,
                    model=usage.model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    embedding_tokens=usage.embedding_tokens,
                    total_tokens=usage.total_tokens,
                    estimated_cost_usd=usage.estimated_cost_usd,
                )
            )
            await session.commit()
        return AnswerRecord(
            id=answer_id,
            collection_id=collection_id,
            question=question,
            answer=answer,
            abstained=abstained,
            citations=list(citations),
            usage=usage,
            created_at=created_at,
        )

    async def get_answer(self, answer_id: str) -> AnswerRecord:
        async with self.sessions() as session:
            answer_row = await session.get(AnswerRow, answer_id)
            if answer_row is None:
                raise NotFoundError("answer not found")
            citation_result = await session.execute(
                select(CitationRow)
                .where(CitationRow.answer_id == answer_id)
                .order_by(CitationRow.rank)
            )
            usage_row = await session.get(UsageRow, answer_id)
        if usage_row is None:
            raise RepositoryError("answer usage is missing")
        citations = [
            Citation(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                quote=row.quote,
                score=row.score,
                rank=row.rank,
            )
            for row in citation_result.scalars().all()
        ]
        usage = Usage(
            provider=usage_row.provider,
            model=usage_row.model,
            prompt_tokens=usage_row.prompt_tokens,
            completion_tokens=usage_row.completion_tokens,
            embedding_tokens=usage_row.embedding_tokens,
            total_tokens=usage_row.total_tokens,
            estimated_cost_usd=usage_row.estimated_cost_usd,
        )
        return AnswerRecord(
            id=answer_row.id,
            collection_id=answer_row.collection_id,
            question=answer_row.question,
            answer=answer_row.answer,
            abstained=answer_row.abstained,
            citations=citations,
            usage=usage,
            created_at=answer_row.created_at,
        )

    async def create_feedback(
        self, answer_id: str, rating: str, comment: str | None
    ) -> FeedbackRecord:
        row = FeedbackRow(
            id=str(uuid4()),
            answer_id=answer_id,
            rating=rating,
            comment=comment,
            created_at=utc_now(),
        )
        async with self.sessions() as session:
            if await session.get(AnswerRow, answer_id) is None:
                raise NotFoundError("answer not found")
            session.add(row)
            await session.commit()
        return FeedbackRecord(
            id=row.id,
            answer_id=row.answer_id,
            rating=row.rating,  # type: ignore[arg-type]
            comment=row.comment,
            created_at=row.created_at,
        )
