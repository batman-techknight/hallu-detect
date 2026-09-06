"""Central configuration. All values overridable via environment variables / .env."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- API ---
    app_name: str = "hallucination-guard"
    cors_origins: list[str] = ["http://localhost:5173"]

    # --- Postgres ---
    database_url: str = "postgresql+asyncpg://hguard:hguard@localhost:5432/hguard"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 60 * 60 * 24  # 1 day for claim->verdict cache

    # --- Model serving ---
    # Point these at a local vLLM OpenAI-compatible server, or a free-tier
    # hosted endpoint (HF Inference Endpoints / Groq / OpenRouter free models).
    generator_base_url: str = "http://localhost:8001/v1"       # model under test
    generator_model: str = "Qwen/Qwen2.5-7B-Instruct"

    judge_base_url: str = "http://localhost:8002/v1"           # different family than generator
    judge_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    claim_extractor_model: str = "Qwen/Qwen2.5-3B-Instruct"    # small/cheap, can share server

    # --- Embeddings / NLI (run locally via HF, free & offline) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    # --- Pipeline thresholds ---
    semantic_entropy_samples: int = 6
    semantic_entropy_temperature: float = 0.9
    entropy_flag_threshold: float = 0.45      # above -> route to retrieval+judge
    nli_contradiction_threshold: float = 0.5  # NLI contradiction prob -> flag claim
    judge_confidence_threshold: float = 0.6   # below -> mark "unverifiable"

    # --- Retrieval ---
    faiss_index_path: str = "./data/faiss.index"
    retrieval_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
