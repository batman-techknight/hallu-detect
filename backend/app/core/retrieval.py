"""FAISS-backed evidence store for retrieval-grounded claim verification.

Free & local: sentence-transformers embeddings + FAISS flat/IVF index. Swap
`add_documents` source for whatever trusted corpus you're grounding against
(internal KB, Wikipedia dump, docs, etc).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import faiss
import numpy as np

from app.config import get_settings
from app.core.nli import _load_embedder

settings = get_settings()


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


class EvidenceStore:
    def __init__(self, index_path: str | None = None):
        self.index_path = index_path or settings.faiss_index_path
        self.meta_path = self.index_path + ".meta.json"
        self._embedder = _load_embedder()
        self._dim = self._embedder.get_sentence_embedding_dimension()
        self._index: faiss.Index
        self._meta: list[dict]
        self._load_or_init()

    def _load_or_init(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self._index = faiss.read_index(self.index_path)
            with open(self.meta_path) as f:
                self._meta = json.load(f)
        else:
            self._index = faiss.IndexFlatIP(self._dim)  # cosine via normalized vectors
            self._meta = []

    def add_documents(self, texts: list[str], sources: list[str]) -> None:
        vecs = self._embedder.encode(texts, normalize_embeddings=True)
        self._index.add(np.asarray(vecs, dtype="float32"))
        self._meta.extend({"text": t, "source": s} for t, s in zip(texts, sources, strict=True))

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        faiss.write_index(self._index, self.index_path)
        with open(self.meta_path, "w") as f:
            json.dump(self._meta, f)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or settings.retrieval_top_k
        if self._index.ntotal == 0:
            return []
        qvec = self._embedder.encode([query], normalize_embeddings=True)
        scores, idxs = self._index.search(np.asarray(qvec, dtype="float32"), min(top_k, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], idxs[0], strict=True):
            if idx == -1:
                continue
            m = self._meta[idx]
            results.append(RetrievedChunk(text=m["text"], source=m["source"], score=float(score)))
        return results
