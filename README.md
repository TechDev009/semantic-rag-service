# Semantic RAG Service

A standalone semantic search and retrieval-augmented generation (RAG) API. The
checkout is useful with no credentials: SQLite, a deterministic hashed embedding
provider, and a deterministic fake JSON LLM are the defaults. Production can use
PostgreSQL/pgvector and OpenAI-compatible adapters without changing the API.

## Architecture

```text
FastAPI -> RagService -> provider protocols
                  |-> EmbeddingProvider (local hash or OpenAI embeddings)
                  |-> GenerationProvider (deterministic fake or OpenAI JSON schema)
                  `-> SQLAlchemy repository
                       |-> SQLite JSON vectors + NumPy hybrid reranking (local)
                       `-> PostgreSQL vector column + HNSW candidates (production)
```

The source uses a `src/` layout and typed Pydantic v2 domain contracts. Documents
are split into bounded chunks, embedded once, and stored with the embedding model,
version, and dimensions. Answers have normalized citation and usage rows; feedback
is stored separately. No LangChain dependency is required.

## Local quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
cd apps/04-semantic-rag-service
uv sync --dev
uv run uvicorn semantic_rag.main:app --reload
```

The default database is `./semantic_rag.db`. Try the API:

```bash
collection=$(curl -s http://localhost:8000/v1/collections \
  -H 'content-type: application/json' \
  -d '{"name":"handbook","description":"local demo"}')
collection_id=$(printf '%s' "$collection" | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')

curl -s http://localhost:8000/v1/documents \
  -H 'content-type: application/json' \
  -d "{\"collection_id\":\"$collection_id\",\"title\":\"Support policy\",\"text\":\"Refunds are available within 30 days of purchase. Contact support with your receipt.\"}"

curl -s http://localhost:8000/v1/answers \
  -H 'content-type: application/json' \
  -d "{\"collection_id\":\"$collection_id\",\"query\":\"How long are refunds available?\"}"
```

Health endpoints are `GET /v1/health/live` and `GET /v1/health/ready`.

## API surface

- `POST /v1/collections` creates an embedding-compatible collection.
- `POST /v1/documents` accepts either `text` or an explicit `chunks` array. Text is
  split at word boundaries with bounded overlap.
- `POST /v1/search` accepts `{collection_id, query, top_k}` and returns ranked
  chunks with vector, keyword, and combined scores.
- `POST /v1/answers` accepts the same fields and returns an answer, abstention
  flag, citations, and token/cost usage.
- `POST /v1/answers/{id}/feedback` accepts `{"rating":"positive"|"negative"}`
  and an optional comment.

All request models reject unknown fields and bound query, chunk, context, and
feedback sizes. Missing collections return 404; duplicate names return 409.

## Provider configuration

Copy `.env.example` to `.env`. Defaults are:

```dotenv
DATABASE_URL=sqlite+aiosqlite:///./semantic_rag.db
EMBEDDING_PROVIDER=local
GENERATION_PROVIDER=fake
EMBEDDING_DIMENSIONS=64
```

For production, set `DATABASE_URL` to a PostgreSQL URL and install/run a
pgvector-enabled PostgreSQL instance. The application creates the `vector`
extension and an HNSW cosine index during initialization. The native vector
adapter currently uses the default 64 dimensions; changing dimensions requires
a matching database migration and vector index.

OpenAI-compatible adapters are opt-in and use the official `openai.AsyncOpenAI`
client:

```dotenv
DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5432/rag
EMBEDDING_PROVIDER=openai
GENERATION_PROVIDER=openai
OPENAI_API_KEY=replace-me
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_GENERATION_MODEL=gpt-4o-mini
# OPENAI_BASE_URL=https://your-compatible-endpoint/v1
INPUT_COST_PER_1K_TOKENS=0.00015
OUTPUT_COST_PER_1K_TOKENS=0.0006
```

Provider protocols are in `src/semantic_rag/providers.py`; a custom adapter only
needs `embed()` or `generate()` and the declared model metadata. The generation
adapter requests strict JSON schema output. The service validates it again,
rejects malformed output, and never trusts provider-supplied citation IDs that
are not in the retrieved set.

## RAG safety and threat model

- **Prompt injection in documents:** documents are quoted JSON data and the system
  prompt explicitly says not to execute their instructions. The fake provider
  only reads data fields and filters common injection phrases. A real model is
  still treated as untrusted and its output is schema- and citation-validated.
- **Citation fabrication:** an answer is accepted only when it has non-empty IDs
  that exactly identify retrieved chunks. Quotes come from storage, not from the
  model. Invalid or uncited output becomes a persisted abstention.
- **Insufficient evidence:** retrieval has a minimum score and a lexical/high
  semantic evidence gate. The answer path abstains instead of filling gaps.
- **Data exfiltration:** prompts contain only bounded retrieved text and the
  question. Secrets are not read from documents or sent to providers by this
  service; deployers should still apply provider/network policy.
- **Denial of service:** Pydantic bounds request sizes, chunk count, top-k, and
  context characters. Add authentication, rate limits, quotas, and request
  timeouts at the edge before exposing this service publicly.
- **Tenant isolation:** collection IDs scope every retrieval query. This demo has
  no authentication or authorization layer, so production must add one.

This is a reference implementation, not a complete security boundary. Validate
uploaded content, protect the database, redact logs, and review model/provider
retention settings before production use.

## Evaluation

The small deterministic dataset is in `evaluation/dataset.json`. Run:

```bash
uv run python scripts/evaluate.py
```

The script reports retrieval recall, that every returned citation belongs to its
retrieved set, and that an unsupported question abstains. It exits non-zero if
those invariants fail. The same behaviors are covered by offline pytest tests.

## Database and migrations

`alembic/` contains the migration entry point. Local demos initialize tables
lazily; for a managed database use `uv run alembic upgrade head` after setting
`DATABASE_URL`. SQLite uses JSON embeddings and NumPy cosine similarity. The
PostgreSQL path uses pgvector cosine distance for HNSW candidate retrieval and a
small keyword rerank, which is a pragmatic hybrid path rather than a full
language-specific PostgreSQL text search configuration.

## Development

```bash
uv run pytest
uv run ruff format --check src tests scripts
uv run ruff check src tests scripts
uv run pyright
```

Tests never call paid providers. Docker examples are provided in
`docker-compose.yml`; CI runs the offline checks only.

## Limitations

The local embedding model is a stable feature hash, not a learned semantic
model, so synonym recall is limited. Token counts are conservative estimates
unless a provider returns usage. There is no authentication, streaming,
background ingestion, document deletion, OCR, ACL filtering, reranking model,
or prompt/tokenizer-specific budget accounting. PostgreSQL deployments should
use Alembic migrations, connection pooling, monitoring, and a dimension-specific
vector migration before changing embedding models.
