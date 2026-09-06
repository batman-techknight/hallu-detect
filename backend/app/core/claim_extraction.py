"""Decompose a free-text LLM answer into atomic, independently-checkable
factual claims using a small/cheap instruct model. Structured JSON output
only -- no free text -- so downstream steps can parse it deterministically.
"""
from __future__ import annotations

import json

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

settings = get_settings()

EXTRACTION_SYSTEM_PROMPT = """You decompose text into atomic factual claims.
Rules:
- Each claim must be a single, self-contained, checkable statement of fact.
- Do not include opinions, hedges, or the question itself.
- Resolve pronouns to their referents where possible.
- Respond with ONLY a JSON array of strings. No preamble, no markdown fences.
Example output: ["The Eiffel Tower was completed in 1889.", "It is located in Paris, France."]
"""


class ExtractedClaims(BaseModel):
    claims: list[str]


class ClaimExtractor:
    def __init__(self, http_client: httpx.AsyncClient | None = None):
        self._client = http_client or httpx.AsyncClient(base_url=settings.generator_base_url, timeout=60)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def extract(self, text: str) -> ExtractedClaims:
        resp = await self._client.post(
            "/chat/completions",
            json={
                "model": settings.claim_extractor_model,
                "messages": [
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.0,
                "max_tokens": 512,
            },
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(raw)
            return ExtractedClaims(claims=parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Claim extractor returned unparseable output: {raw!r}") from e

    async def aclose(self) -> None:
        await self._client.aclose()
