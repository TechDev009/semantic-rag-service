from collections.abc import Callable, Coroutine
from pathlib import Path
from runpy import run_path
from typing import Any, cast

import pytest


@pytest.mark.asyncio
async def test_evaluation_fixture_invariants() -> None:
    namespace = run_path(str(Path(__file__).parents[1] / "scripts/evaluate.py"))
    evaluate = cast(
        Callable[[], Coroutine[Any, Any, dict[str, float | int]]], namespace["evaluate"]
    )
    metrics = await evaluate()
    assert metrics["retrieval_recall_at_3"] == 1.0
    assert metrics["valid_citation_answers"] == metrics["total_questions"]
    assert metrics["abstention_passes"] == 1
