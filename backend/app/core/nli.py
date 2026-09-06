"""Local, free, offline NLI scoring + semantic similarity.

Uses a HuggingFace cross-encoder NLI model (no API cost, runs on CPU or GPU)
to score entailment/neutral/contradiction between a claim and a piece of
evidence, and a sentence-transformers bi-encoder for fast semantic similarity
(used as a pre-filter before the more expensive cross-encoder pass).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util

from app.config import get_settings

settings = get_settings()


@dataclass
class NLIVerdict:
    entailment: float
    neutral: float
    contradiction: float

    @property
    def label(self) -> str:
        best = max(("entailment", self.entailment), ("neutral", self.neutral),
                    ("contradiction", self.contradiction), key=lambda kv: kv[1])
        return best[0]


@lru_cache
def _load_cross_encoder() -> CrossEncoder:
    return CrossEncoder(settings.nli_model)


@lru_cache
def _load_embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


class NLIScorer:
    """Thin wrapper so this can be mocked easily in tests."""

    def __init__(self) -> None:
        self._ce = _load_cross_encoder()
        self._embedder = _load_embedder()

    def score(self, premise: str, hypothesis: str) -> NLIVerdict:
        logits = self._ce.predict([(premise, hypothesis)])[0]
        probs = torch.softmax(torch.tensor(logits), dim=-1).tolist()
        # cross-encoder/nli-deberta-v3-base label order: contradiction, entailment, neutral
        contradiction, entailment, neutral = probs
        return NLIVerdict(entailment=entailment, neutral=neutral, contradiction=contradiction)

    def entailment_prob(self, premise: str, hypothesis: str) -> float:
        return self.score(premise, hypothesis).entailment

    def similarity(self, a: str, b: str) -> float:
        emb = self._embedder.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
        return float(util.cos_sim(emb[0], emb[1]))

    def rank_by_similarity(self, query: str, candidates: list[str]) -> list[tuple[str, float]]:
        q = self._embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        c = self._embedder.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(q, c)[0].tolist()
        return sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)
