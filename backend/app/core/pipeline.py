"""Orchestrates the full detection pipeline:

  1. Semantic entropy (cheap triage signal, optional)
  2. Claim extraction (decompose response into atomic claims)
  3. Retrieval + NLI (grounded verification per claim, cheap-ish)
  4. LLM judge (expensive, only for claims Layer 3 couldn't confidently resolve)

Results are cached per-claim in Redis so repeated factual claims across
different user queries skip straight to a cached verdict.
"""
from __future__ import annotations

from app.cache.redis_client import (
    get_cached_request,
    get_cached_verdict,
    set_cached_request,
    set_cached_verdict,
)
from app.config import get_settings
from app.core.claim_extraction import ClaimExtractor
from app.core.judge import JudgeLabel, JudgeVerdict, LLMJudge
from app.core.nli import NLIScorer
from app.core.retrieval import EvidenceStore
from app.core.semantic_entropy import SemanticEntropyEstimator
from app.schemas.verify import ClaimVerdict, VerifyRequest, VerifyResponse

settings = get_settings()


class HallucinationPipeline:
    def __init__(
        self,
        entropy_estimator: SemanticEntropyEstimator | None = None,
        extractor: ClaimExtractor | None = None,
        evidence_store: EvidenceStore | None = None,
        nli: NLIScorer | None = None,
        judge: LLMJudge | None = None,
    ):
        self.nli = nli or NLIScorer()
        self.entropy_estimator = entropy_estimator or SemanticEntropyEstimator(nli=self.nli)
        self.extractor = extractor or ClaimExtractor()
        self.store = evidence_store or EvidenceStore()
        self.judge = judge or LLMJudge()

    async def _verify_claim(self, claim: str) -> ClaimVerdict:
        cached = await get_cached_verdict(claim)
        if cached:
            return ClaimVerdict(**cached, from_cache=True)

        # Layer 2: retrieval + NLI
        chunks = self.store.search(claim)
        evidence_texts = [c.text for c in chunks]

        best_entail, best_contra = 0.0, 0.0
        for ev in evidence_texts:
            v = self.nli.score(premise=ev, hypothesis=claim)
            best_entail = max(best_entail, v.entailment)
            best_contra = max(best_contra, v.contradiction)

        needs_judge = (
            not evidence_texts
            or best_contra >= settings.nli_contradiction_threshold
            or (best_entail < 0.6 and best_contra < 0.5)  # ambiguous
        )

        if needs_judge:
            judge_verdict: JudgeVerdict = await self.judge.judge(claim, evidence_texts)
            label = judge_verdict.label
            confidence = judge_verdict.confidence
            rationale = judge_verdict.rationale
            citation = judge_verdict.citation
            if confidence < settings.judge_confidence_threshold:
                label = JudgeLabel.UNVERIFIABLE
        else:
            label = JudgeLabel.SUPPORTED if best_entail >= best_contra else JudgeLabel.CONTRADICTED
            confidence = max(best_entail, best_contra)
            rationale = "Resolved by retrieval + NLI without judge escalation."
            citation = evidence_texts[0] if evidence_texts else None

        verdict = ClaimVerdict(
            claim=claim,
            label=label,
            confidence=confidence,
            rationale=rationale,
            citation=citation,
            evidence=evidence_texts,
            from_cache=False,
        )
        await set_cached_verdict(claim, verdict.model_dump(exclude={"from_cache"}))
        return verdict

    @staticmethod
    def _risk_score(entropy: float | None, claim_verdicts: list[ClaimVerdict]) -> float:
        if not claim_verdicts:
            return entropy or 0.0
        bad = sum(
            1
            for c in claim_verdicts
            if c.label in (JudgeLabel.CONTRADICTED, JudgeLabel.UNVERIFIABLE)
        )
        claim_risk = bad / len(claim_verdicts)
        if entropy is None:
            return claim_risk
        # weighted blend: claim-level evidence dominates, entropy nudges it
        return round(min(1.0, 0.7 * claim_risk + 0.3 * entropy), 4)

    async def run(self, req: VerifyRequest) -> VerifyResponse:
        # Request-level cache: identical (prompt, response) pairs skip the
        # whole pipeline — no entropy sampling, extraction, retrieval, or
        # judge calls at all. This is the biggest cost/latency win since it
        # short-circuits before any model is touched.
        cached = await get_cached_request(req.prompt, req.response)
        if cached:
            return VerifyResponse(**cached)

        entropy_val: float | None = None
        entropy_flagged = False

        if req.run_semantic_entropy:
            result = await self.entropy_estimator.estimate(req.prompt)
            entropy_val = round(result.entropy, 4)
            entropy_flagged = result.is_uncertain

        extracted = await self.extractor.extract(req.response)
        claim_verdicts = [await self._verify_claim(c) for c in extracted.claims]

        risk = self._risk_score(entropy_val, claim_verdicts)

        result = VerifyResponse(
            prompt=req.prompt,
            response=req.response,
            semantic_entropy=entropy_val,
            entropy_flagged=entropy_flagged,
            claims=claim_verdicts,
            overall_risk_score=risk,
        )
        await set_cached_request(req.prompt, req.response, result.model_dump())
        return result

    async def aclose(self) -> None:
        await self.entropy_estimator.aclose()
        await self.extractor.aclose()
        await self.judge.aclose()
