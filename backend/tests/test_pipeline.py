from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.claim_extraction import ExtractedClaims
from app.core.judge import JudgeLabel, JudgeVerdict
from app.core.pipeline import HallucinationPipeline
from app.core.retrieval import RetrievedChunk
from app.schemas.verify import VerifyRequest


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr("app.core.pipeline.get_cached_verdict", AsyncMock(return_value=None))
    monkeypatch.setattr("app.core.pipeline.set_cached_verdict", AsyncMock())
    monkeypatch.setattr("app.core.pipeline.get_cached_request", AsyncMock(return_value=None))
    monkeypatch.setattr("app.core.pipeline.set_cached_request", AsyncMock())

    entropy_estimator = MagicMock()
    entropy_estimator.estimate = AsyncMock(
        return_value=MagicMock(entropy=0.8, is_uncertain=True)
    )
    entropy_estimator.aclose = AsyncMock()

    extractor = MagicMock()
    extractor.extract = AsyncMock(
        return_value=ExtractedClaims(claims=["The Eiffel Tower was built in 1889."])
    )
    extractor.aclose = AsyncMock()

    store = MagicMock()
    store.search.return_value = [
        RetrievedChunk(text="The Eiffel Tower was completed in 1889.", source="wiki", score=0.9)
    ]

    nli = MagicMock()
    nli.score.return_value = MagicMock(entailment=0.9, contradiction=0.01, neutral=0.09)

    judge = MagicMock()
    judge.judge = AsyncMock(
        return_value=JudgeVerdict(
            label=JudgeLabel.SUPPORTED, confidence=0.95, rationale="matches evidence", citation=None
        )
    )
    judge.aclose = AsyncMock()

    return HallucinationPipeline(
        entropy_estimator=entropy_estimator,
        extractor=extractor,
        evidence_store=store,
        nli=nli,
        judge=judge,
    )


async def test_run_resolves_supported_claim_without_judge(pipeline):
    req = VerifyRequest(prompt="When was the Eiffel Tower built?", response="It was built in 1889.")
    result = await pipeline.run(req)

    assert result.semantic_entropy == 0.8
    assert result.entropy_flagged is True
    assert len(result.claims) == 1
    assert result.claims[0].label == JudgeLabel.SUPPORTED
    # high entailment, low contradiction -> judge should not be needed
    pipeline.judge.judge.assert_not_called()
    assert 0.0 <= result.overall_risk_score <= 1.0


async def test_risk_score_rises_with_contradicted_claims(pipeline):
    pipeline.nli.score.return_value = MagicMock(entailment=0.1, contradiction=0.9, neutral=0.0)
    req = VerifyRequest(prompt="p", response="r", run_semantic_entropy=False)
    result = await pipeline.run(req)

    assert result.claims[0].label == JudgeLabel.CONTRADICTED
    assert result.overall_risk_score > 0.5


async def test_repeated_request_short_circuits_via_cache(pipeline, monkeypatch):
    cached_payload = {
        "prompt": "p",
        "response": "r",
        "semantic_entropy": None,
        "entropy_flagged": False,
        "claims": [],
        "overall_risk_score": 0.1,
    }
    monkeypatch.setattr(
        "app.core.pipeline.get_cached_request", AsyncMock(return_value=cached_payload)
    )
    req = VerifyRequest(prompt="p", response="r", run_semantic_entropy=False)
    result = await pipeline.run(req)

    assert result.overall_risk_score == 0.1
    pipeline.extractor.extract.assert_not_called()
    pipeline.entropy_estimator.estimate.assert_not_called()
