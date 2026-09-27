"""ASGI entry point and local command runner."""

import uvicorn

from .api import create_app
from .config import get_settings

app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "semantic_rag.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
