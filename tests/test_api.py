import httpx
import pytest

from semantic_rag.api import create_app


@pytest.mark.asyncio
async def test_api_validation_search_answer_and_feedback(rag_components) -> None:
    settings, database, service = rag_components
    app = create_app(
        settings=settings,
        database=database,
        embeddings=service.embeddings,
        generation=service.generation,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        live = await client.get("/v1/health/live")
        assert live.status_code == 200
        assert live.json() == {"status": "ok"}

        invalid = await client.post("/v1/documents", json={"title": "missing content"})
        assert invalid.status_code == 422

        created = await client.post(
            "/v1/collections", json={"name": "api-demo", "description": "test"}
        )
        assert created.status_code == 201
        collection_id = created.json()["id"]

        document = await client.post(
            "/v1/documents",
            json={
                "collection_id": collection_id,
                "title": "Refund policy",
                "chunks": ["The refund window is 30 days.", "Receipts are required."],
            },
        )
        assert document.status_code == 201
        assert document.json()["chunk_count"] == 2

        search = await client.post(
            "/v1/search",
            json={
                "collection_id": collection_id,
                "query": "What is the refund window?",
                "top_k": 2,
            },
        )
        assert search.status_code == 200
        assert search.json()["hits"]

        answer = await client.post(
            "/v1/answers",
            json={"collection_id": collection_id, "query": "What is the refund window?"},
        )
        assert answer.status_code == 200
        payload = answer.json()
        assert payload["abstained"] is False
        assert payload["citations"]
        assert payload["usage"]["total_tokens"] > 0

        feedback = await client.post(
            f"/v1/answers/{payload['id']}/feedback",
            json={"rating": "positive", "comment": "grounded"},
        )
        assert feedback.status_code == 201
        assert feedback.json()["answer_id"] == payload["id"]

        ready = await client.get("/v1/health/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_api_rejects_unknown_fields_and_missing_resources(rag_components) -> None:
    settings, database, service = rag_components
    app = create_app(
        settings=settings,
        database=database,
        embeddings=service.embeddings,
        generation=service.generation,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        extra = await client.post("/v1/collections", json={"name": "x", "unexpected": True})
        assert extra.status_code == 422
        missing = await client.post(
            "/v1/search", json={"collection_id": "missing", "query": "anything"}
        )
        assert missing.status_code == 404
