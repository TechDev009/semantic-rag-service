# Contributing

## Setup

```bash
uv sync --dev
```

The default tests use deterministic providers and an in-memory SQLite database.
Do not add tests that call paid or network services.

## Checks

Run before opening a change:

```bash
uv run pytest
uv run ruff format --check src tests scripts
uv run ruff check src tests scripts
uv run pyright
```

Use the existing `EmbeddingProvider` and `GenerationProvider` protocols for new
providers. Keep provider-specific SDK imports behind adapters and keep the fake
providers deterministic. New answer behavior should include tests for malformed
output, unsupported questions, citations, prompt-injection-like document text,
and usage/cost accounting.

## Pull requests

Explain the API or schema impact, migration requirements, security implications,
and any new configuration. Keep commits focused and update README, evaluation
fixtures, and tests when behavior changes.
