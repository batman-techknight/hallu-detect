from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.pipeline import HallucinationPipeline
from app.db.models import ClaimRecord, VerificationRun
from app.db.session import get_db
from app.schemas.verify import VerifyRequest, VerifyResponse

router = APIRouter(prefix="/verify", tags=["verify"])

_pipeline = HallucinationPipeline()  # module-level singleton; models load once


@router.post("", response_model=VerifyResponse)
async def verify(req: VerifyRequest, db: AsyncSession = Depends(get_db)) -> VerifyResponse:
    result = await _pipeline.run(req)

    run = VerificationRun(
        prompt=result.prompt,
        response=result.response,
        semantic_entropy=result.semantic_entropy,
        overall_risk_score=result.overall_risk_score,
    )
    run.claims = [
        ClaimRecord(
            claim=c.claim,
            label=c.label.value,
            confidence=c.confidence,
            rationale=c.rationale,
            citation=c.citation,
            evidence=c.evidence,
        )
        for c in result.claims
    ]
    db.add(run)
    await db.commit()

    return result
