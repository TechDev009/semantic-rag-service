"""Typed application configuration.

The defaults intentionally use SQLite and deterministic local providers, so a new
checkout can be exercised without credentials or an external service.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and an optional .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "semantic-rag-service"
    environment: str = "development"
    log_level: str = "INFO"
    database_url: str = "sqlite+aiosqlite:///./semantic_rag.db"

    embedding_provider: str = "local"
    embedding_model: str = "local-hash-v1"
    embedding_dimensions: int = Field(default=64, ge=8, le=4096)

    generation_provider: str = "fake"
    generation_model: str = "fake-rag-v1"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_generation_model: str = "gpt-4o-mini"

    retrieval_default_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_max_top_k: int = Field(default=20, ge=1, le=100)
    retrieval_min_score: float = Field(default=0.18, ge=-1.0, le=1.0)
    max_context_chars: int = Field(default=6000, ge=1000, le=50000)
    max_chunk_chars: int = Field(default=1200, ge=100, le=10000)
    chunk_overlap_chars: int = Field(default=120, ge=0, le=1000)
    max_question_chars: int = Field(default=4000, ge=20, le=20000)

    input_cost_per_1k_tokens: float = Field(default=0.0, ge=0.0)
    output_cost_per_1k_tokens: float = Field(default=0.0, ge=0.0)

    @property
    def async_database_url(self) -> str:
        """Return a SQLAlchemy async URL for common Postgres URL spellings."""

        if self.database_url.startswith("postgresql://"):
            return "postgresql+asyncpg://" + self.database_url.removeprefix("postgresql://")
        if self.database_url.startswith("postgres://"):
            return "postgresql+asyncpg://" + self.database_url.removeprefix("postgres://")
        return self.database_url

    @property
    def is_postgres(self) -> bool:
        return self.async_database_url.startswith("postgresql+")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings object used by the CLI application."""

    return Settings()
