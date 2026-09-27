import pytest

from semantic_rag.db import NotFoundError


@pytest.mark.asyncio
async def test_ingestion_persists_chunks_and_hybrid_search(rag_components) -> None:
    _, database, service = rag_components
    collection = await database.create_collection(
        "support", "support docs", service.embeddings.model, service.embeddings.dimensions
    )
    document, chunk_ids = await service.ingest_document(
        collection.id,
        "Refund policy",
        "policy.md",
        {"team": "support"},
        "Refunds are available for 30 days. Receipts are required.",
        None,
    )

    assert document.collection_id == collection.id
    assert len(chunk_ids) == 1
    query_vector = (await service.embeddings.embed(["How long are refunds available?"]))[0]
    hits = await database.search(collection.id, query_vector, "How long are refunds available?", 3)
    assert hits
    assert hits[0].chunk_id == chunk_ids[0]
    assert hits[0].keyword_score > 0
    assert hits[0].title == "Refund policy"


@pytest.mark.asyncio
async def test_unknown_collection_is_not_cross_tenant(rag_components) -> None:
    _, database, service = rag_components
    query_vector = (await service.embeddings.embed(["anything"]))[0]
    with pytest.raises(NotFoundError):
        await database.search("missing", query_vector, "anything", 5)
