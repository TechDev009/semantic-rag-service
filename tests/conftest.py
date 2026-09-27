from collections.abc import AsyncIterator

import pytest_asyncio

from semantic_rag.config import Settings
from semantic_rag.db import Database
from semantic_rag.providers import DeterministicEmbeddingProvider, DeterministicFakeLLM
from semantic_rag.rag import RagService


@pytest_asyncio.fixture
async def rag_components() -> AsyncIterator[tuple[Settings, Database, RagService]]:
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        retrieval_min_score=0.1,
        input_cost_per_1k_tokens=0.01,
        output_cost_per_1k_tokens=0.02,
    )
    database = Database(settings)
    embeddings = DeterministicEmbeddingProvider(settings.embedding_dimensions)
    generation = DeterministicFakeLLM()
    service = RagService(database, embeddings, generation, settings)
    await database.initialize()
    try:
        yield settings, database, service
    finally:
        await database.dispose()
