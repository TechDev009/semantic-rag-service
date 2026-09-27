FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.5.14 /uv /uvx /bin/
COPY pyproject.toml uv.lock* README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["uv", "run", "--no-dev", "uvicorn", "semantic_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
