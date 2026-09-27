"""Run the small deterministic retrieval/RAG evaluation set."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from semantic_rag.config import Settings
from semantic_rag.db import Database
from semantic_rag.providers import DeterministicEmbeddingProvider, DeterministicFakeLLM
from semantic_rag.rag import RagService


async def evaluate() -> dict[str, float | int]:
    fixture = json.loads((Path(__file__).parents[1] / "evaluation/dataset.json").read_text())
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        retrieval_min_score=0.12,
        input_cost_per_1k_tokens=0.001,
        output_cost_per_1k_tokens=0.002,
    )
    database = Database(settings)
    embeddings = DeterministicEmbeddingProvider(settings.embedding_dimensions)
    generation = DeterministicFakeLLM()
    service = RagService(database, embeddings, generation, settings)
    await database.initialize()
    try:
        collection = await database.create_collection(
            "evaluation", "deterministic evaluation", embeddings.model, embeddings.dimensions
        )
        document_ids: dict[str, str] = {}
        for document in fixture["documents"]:
            stored, _ = await service.ingest_document(
                collection.id,
                document["title"],
                document["id"],
                {},
                document["text"],
                None,
            )
            document_ids[document["id"]] = stored.id

        answerable = 0
        recalled = 0
        valid_citations = 0
        abstention_passed = 0
        for item in fixture["questions"]:
            vector = (await embeddings.embed([item["question"]]))[0]
            hits = await database.search(collection.id, vector, item["question"], 3)
            if item["answerable"]:
                answerable += 1
                expected_document = document_ids[item["document_id"]]
                if any(hit.document_id == expected_document for hit in hits):
                    recalled += 1
            answer = await service.answer(collection.id, item["question"], 3)
            hit_ids = {hit.chunk_id for hit in hits}
            if all(citation.chunk_id in hit_ids for citation in answer.citations):
                valid_citations += 1
            if not item["answerable"] and answer.abstained:
                abstention_passed += 1

        metrics: dict[str, float | int] = {
            "retrieval_recall_at_3": recalled / answerable if answerable else 0.0,
            "answerable_questions": answerable,
            "valid_citation_answers": valid_citations,
            "abstention_passes": abstention_passed,
            "total_questions": len(fixture["questions"]),
        }
        if metrics["retrieval_recall_at_3"] < 1.0 or valid_citations != len(fixture["questions"]):
            raise SystemExit(f"evaluation failed: {metrics}")
        if abstention_passed != 1:
            raise SystemExit(f"abstention evaluation failed: {metrics}")
        return metrics
    finally:
        await database.dispose()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(evaluate()), indent=2, sort_keys=True))
