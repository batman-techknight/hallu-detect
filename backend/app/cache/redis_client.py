"""Redis cache for claim-level verdicts. Same factual claims recur across
many user queries, so caching claim -> verdict avoids re-running retrieval
and (especially) the expensive judge LLM call for repeats.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import redis.asyncio as redis

from app.config import get_settings

settings = get_settings()


def claim_cache_key(claim: str) -> str:
    digest = hashlib.sha256(claim.strip().lower().encode()).hexdigest()
    return f"claim_verdict:{digest}"


def request_cache_key(prompt: str, response: str) -> str:
    """Cache key for a full /verify request, so an identical (prompt, response)
    pair — e.g. a retried request, or the same chatbot answer served to two
    users — skips the entire pipeline (entropy sampling, extraction,
    retrieval, judge) and returns instantly.
    """
    digest = hashlib.sha256(f"{prompt.strip().lower()}||{response.strip().lower()}".encode()).hexdigest()
    return f"verify_request:{digest}"


async def get_cached_request(prompt: str, response: str) -> dict | None:
    r = get_redis()
    raw = await r.get(request_cache_key(prompt, response))
    return json.loads(raw) if raw else None


async def set_cached_request(prompt: str, response: str, result: dict) -> None:
    r = get_redis()
    await r.set(request_cache_key(prompt, response), json.dumps(result), ex=settings.cache_ttl_seconds)


@lru_cache
def get_redis() -> redis.Redis:
    return redis.from_url(settings.redis_url, decode_responses=True)


async def get_cached_verdict(claim: str) -> dict | None:
    r = get_redis()
    raw = await r.get(claim_cache_key(claim))
    return json.loads(raw) if raw else None


async def set_cached_verdict(claim: str, verdict: dict) -> None:
    r = get_redis()
    await r.set(claim_cache_key(claim), json.dumps(verdict), ex=settings.cache_ttl_seconds)
