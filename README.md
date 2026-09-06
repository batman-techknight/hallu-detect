# Hallu-detect

LLM-judges-LLM hallucination detection & correction system. Four-layer pipeline
(semantic entropy → claim extraction → retrieval + NLI → LLM judge), Redis-cached
at both the **request level** (identical prompt+response pairs skip the whole
pipeline) and the **claim level** (repeated factual claims skip retrieval + judge).

## Stack

| Layer | Tech |
|---|---|
| Backend | Python, FastAPI, Pydantic v2 |
| DB | PostgreSQL (async SQLAlchemy + asyncpg) |
| Cache | Redis (request-level + claim-level) |
| Embeddings / NLI | Sentence-Transformers, HuggingFace Transformers (local, free) |
| Vector store | FAISS |
| Generator / Judge LLMs | Qwen2.5 / Llama-3.1, served via vLLM (OpenAI-compatible API) |
| Frontend | React + TypeScript + Vite |
| Tests | pytest, pytest-asyncio |
| Lint/format | ruff |
| Containers | Docker, docker-compose |
| CI | GitHub Actions |

## Folder structure

```
hallu-detect/
├── .github/
│   └── workflows/
│       └── ci.yml                 # lint + test backend, build frontend
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app entrypoint
│   │   ├── config.py              # pydantic-settings (env-driven config)
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── health.py
│   │   │       └── verify.py      # POST /verify
│   │   ├── core/
│   │   │   ├── semantic_entropy.py   # Layer 1: cheap uncertainty signal
│   │   │   ├── claim_extraction.py   # Layer 2a: decompose into atomic claims
│   │   │   ├── retrieval.py          # Layer 2b: FAISS evidence store
│   │   │   ├── nli.py                # NLI + sentence-transformers similarity
│   │   │   ├── judge.py              # Layer 3: LLM-as-judge (structured verdict)
│   │   │   └── pipeline.py           # orchestrator wiring all layers + cache
│   │   ├── cache/
│   │   │   └── redis_client.py       # request-level + claim-level Redis cache
│   │   ├── db/
│   │   │   ├── models.py             # SQLAlchemy models (Postgres)
│   │   │   └── session.py
│   │   └── schemas/
│   │       └── verify.py             # Pydantic request/response models
│   ├── tests/
│   │   ├── test_semantic_entropy.py
│   │   └── test_pipeline.py
│   ├── pyproject.toml
│   ├── ruff.toml
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── api/client.ts             # typed fetch wrapper for /verify
│   │   └── components/ClaimBadge.tsx
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── Dockerfile
├── docker-compose.yml               # postgres + redis + vllm x2 + backend + frontend
├── .env.example
├── .gitignore
└── README.md
```

## Free model choices

- **Generator under test**: `Qwen/Qwen2.5-7B-Instruct` (or any model you're evaluating).
- **Judge**: `meta-llama/Llama-3.1-8B-Instruct` — deliberately a *different family*
  from the generator, so it doesn't share the same blind spots.
- **Claim extractor**: `Qwen/Qwen2.5-3B-Instruct` — small, cheap, runs on the same
  vLLM server as the generator.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` — local, free, fast.
- **NLI**: `cross-encoder/nli-deberta-v3-base` — local, free.
- No paid API keys required. If you don't have a GPU, point `GENERATOR_BASE_URL`
  / `JUDGE_BASE_URL` at a free hosted OpenAI-compatible endpoint instead (HF
  Inference Endpoints free tier, Groq free tier, OpenRouter free models) — the
  code doesn't care, it just calls `/chat/completions`.

## Setup & run

### Option A — everything in Docker (recommended if you have a GPU for vLLM)

```bash
git clone <your-repo-url> hallu-detect
cd hallu-detect
cp .env.example .env          # edit values if needed

docker compose up --build
```

- Backend: http://localhost:8000/docs (FastAPI Swagger UI)
- Frontend: http://localhost:5173
- Postgres: localhost:5432, Redis: localhost:6379

### Option B — local dev (no Docker, no GPU — point at a hosted free endpoint)

**1. Backend**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"

cp ../.env.example .env
# edit .env: set GENERATOR_BASE_URL / JUDGE_BASE_URL to your free endpoint

# Postgres + Redis via Docker even in local dev mode:
docker run -d --name hguard-pg -e POSTGRES_USER=hguard -e POSTGRES_PASSWORD=hguard \
  -e POSTGRES_DB=hguard -p 5432:5432 postgres:16-alpine
docker run -d --name hguard-redis -p 6379:6379 redis:7-alpine

# create tables (simple approach; swap for Alembic migrations in production)
python - <<'PY'
import asyncio
from app.db.models import Base
from app.db.session import engine

async def main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

asyncio.run(main())
PY

uvicorn app.main:app --reload --port 8000
```

**2. Frontend**

```bash
cd frontend
npm install
cp ../.env.example .env.local     # keep only VITE_API_BASE_URL
npm run dev
```

### Running tests & lint

```bash
cd backend
ruff check .            # lint
ruff format .           # auto-format
pytest --cov=app        # unit tests (fully mocked — no network/model calls)
```

### Building your evidence index (RAG corpus)

```python
from app.core.retrieval import EvidenceStore

store = EvidenceStore()
store.add_documents(
    texts=["The Eiffel Tower was completed in 1889.", "..."],
    sources=["wikipedia:eiffel_tower", "..."],
)
store.save()
```

## API

```
POST /verify
{
  "prompt": "When was the Eiffel Tower built?",
  "response": "The Eiffel Tower was built in 1887 by Gustave Eiffel.",
  "run_semantic_entropy": true
}
```

Returns per-claim verdicts (`supported` / `contradicted` / `unverifiable`), the
semantic entropy score, and an overall `overall_risk_score` (0 = trustworthy,
1 = likely hallucinated). Identical `(prompt, response)` pairs are served
straight from Redis on repeat calls.

## Uploading this project to GitHub

From the project root (`hallucination-guard/`):

```bash
git init
git add .
git commit -m "Initial commit: hallu-detect full-stack pipeline"

# Create the repo on GitHub first (via github.com/new, or gh CLI below),
# then link it:
git branch -M main
git remote add origin https://github.com/<your-username>/hallu-detect.git
git push -u origin main
```

Or, using the GitHub CLI (`gh`) to create the repo in one step instead of
using the website:

```bash
gh repo create hallu-detect --public --source=. --remote=origin --push
```

Subsequent updates:

```bash
git add .
git commit -m "Describe your change"
git push
```

GitHub Actions (`.github/workflows/ci.yml`) will automatically lint, test,
and build on every push/PR once the repo is on GitHub — no extra setup needed.
