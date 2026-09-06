"""LLM judge -- the most expensive, most accurate layer. Invoked only on
claims that Layer 1 (semantic entropy) or Layer 2 (retrieval + NLI) flagged
as uncertain, keeping cost bounded. Uses a DIFFERENT model family than the
generator on purpose: same-family judges correlate with the generator's own
blind spots and under-flag its failure modes.
"""
from __future__ import annotations

import json
from enum import StrEnum

import httpx
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

settings = get_settings()


class JudgeLabel(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


class JudgeVerdict(BaseModel):
    label: JudgeLabel
    confidence: float
    rationale: str
    citation: str | None = None


JUDGE_SYSTEM_PROMPT = """You are a strict fact-verification judge. You are given a CLAIM and a
set of EVIDENCE passages. Decide whether the evidence supports, contradicts, or is insufficient
to verify the claim. Be conservative: if evidence doesn't clearly address the claim, say
"unverifiable" rather than guessing.

Respond with ONLY a JSON object of this exact shape, no markdown fences:
{"label": "supported" | "contradicted" | "unverifiable", "confidence": <0..1 float>,
 "rationale": "<one sentence>", "citation": "<short quote/id from evidence or null>"}
"""


class LLMJudge:
    def __init__(self, http_client: httpx.AsyncClient | None = None):
        self._client = http_client or httpx.AsyncClient(base_url=settings.judge_base_url, timeout=60)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def judge(self, claim: str, evidence: list[str]) -> JudgeVerdict:
        evidence_block = "\n".join(f"[{i}] {e}" for i, e in enumerate(evidence)) or "(no evidence retrieved)"
        user_msg = f"CLAIM: {claim}\n\nEVIDENCE:\n{evidence_block}"

        resp = await self._client.post(
            "/chat/completions",
            json={
                "model": settings.judge_model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.0,
                "max_tokens": 256,
            },
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        return JudgeVerdict(**data)

    async def aclose(self) -> None:
        await self._client.aclose()
