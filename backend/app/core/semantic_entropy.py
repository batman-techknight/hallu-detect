"""Semantic entropy (Kuhn et al., 2023 style, simplified).

Idea: sample the generator N times at temperature > 0 for the same prompt.
Cluster the samples by *bidirectional entailment* (two answers are the "same
meaning" if each entails the other under NLI). Entropy over cluster sizes
approximates how uncertain the model actually is about the *meaning* of its
answer, not just the surface tokens. High entropy -> likely hallucination
risk -> route the claim to the more expensive retrieval + judge layers.

This needs zero extra "judge" LLM calls beyond re-sampling the generator,
which makes it the cheapest signal in the pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import httpx

from app.config import get_settings
from app.core.nli import NLIScorer

settings = get_settings()


@dataclass
class SemanticEntropyResult:
    entropy: float          # normalized 0..1 (0 = fully consistent, 1 = maximally split)
    num_clusters: int
    samples: list[str]
    cluster_assignment: list[int]

    @property
    def is_uncertain(self) -> bool:
        return self.entropy >= settings.entropy_flag_threshold


class SemanticEntropyEstimator:
    def __init__(self, nli: NLIScorer | None = None, http_client: httpx.AsyncClient | None = None):
        self.nli = nli or NLIScorer()
        self._client = http_client or httpx.AsyncClient(base_url=settings.generator_base_url, timeout=60)

    async def _sample(self, prompt: str, n: int, temperature: float) -> list[str]:
        """Fire N chat-completions against an OpenAI-compatible endpoint (vLLM serves this)."""
        outputs: list[str] = []
        for _ in range(n):
            resp = await self._client.post(
                "/chat/completions",
                json={
                    "model": settings.generator_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 256,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            outputs.append(data["choices"][0]["message"]["content"].strip())
        return outputs

    def _cluster_by_entailment(self, samples: list[str]) -> list[int]:
        """Greedy clustering: two texts are in the same cluster iff they
        bidirectionally entail each other above the NLI entailment threshold.
        """
        cluster_of: list[int] = [-1] * len(samples)
        reps: list[str] = []  # one representative sample per cluster

        for i, text in enumerate(samples):
            placed = False
            for cid, rep in enumerate(reps):
                fwd = self.nli.entailment_prob(premise=rep, hypothesis=text)
                bwd = self.nli.entailment_prob(premise=text, hypothesis=rep)
                if fwd > 0.5 and bwd > 0.5:
                    cluster_of[i] = cid
                    placed = True
                    break
            if not placed:
                reps.append(text)
                cluster_of[i] = len(reps) - 1
        return cluster_of

    @staticmethod
    def _normalized_entropy(cluster_of: list[int]) -> float:
        n = len(cluster_of)
        if n <= 1:
            return 0.0
        counts: dict[int, int] = {}
        for c in cluster_of:
            counts[c] = counts.get(c, 0) + 1
        probs = [c / n for c in counts.values()]
        raw = -sum(p * math.log(p) for p in probs)
        max_possible = math.log(n)  # entropy if every sample were its own cluster
        return raw / max_possible if max_possible > 0 else 0.0

    async def estimate(
        self,
        prompt: str,
        n: int | None = None,
        temperature: float | None = None,
    ) -> SemanticEntropyResult:
        n = n or settings.semantic_entropy_samples
        temperature = temperature if temperature is not None else settings.semantic_entropy_temperature

        samples = await self._sample(prompt, n=n, temperature=temperature)
        cluster_of = self._cluster_by_entailment(samples)
        entropy = self._normalized_entropy(cluster_of)

        return SemanticEntropyResult(
            entropy=entropy,
            num_clusters=len(set(cluster_of)),
            samples=samples,
            cluster_assignment=cluster_of,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
