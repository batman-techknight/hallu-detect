from pydantic import BaseModel, Field

from app.core.judge import JudgeLabel


class VerifyRequest(BaseModel):
    prompt: str = Field(..., description="Original user prompt sent to the generator")
    response: str = Field(..., description="LLM-generated answer to check for hallucinations")
    run_semantic_entropy: bool = Field(
        default=True, description="Re-sample the generator to estimate semantic entropy"
    )


class ClaimVerdict(BaseModel):
    claim: str
    label: JudgeLabel
    confidence: float
    rationale: str
    citation: str | None = None
    evidence: list[str] = Field(default_factory=list)
    from_cache: bool = False


class VerifyResponse(BaseModel):
    prompt: str
    response: str
    semantic_entropy: float | None = None
    entropy_flagged: bool = False
    claims: list[ClaimVerdict]
    overall_risk_score: float = Field(..., description="0 (trustworthy) .. 1 (likely hallucinated)")
