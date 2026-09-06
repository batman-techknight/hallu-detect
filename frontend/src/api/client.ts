export type JudgeLabel = "supported" | "contradicted" | "unverifiable";

export interface ClaimVerdict {
  claim: string;
  label: JudgeLabel;
  confidence: number;
  rationale: string;
  citation: string | null;
  evidence: string[];
  from_cache: boolean;
}

export interface VerifyResponse {
  prompt: string;
  response: string;
  semantic_entropy: number | null;
  entropy_flagged: boolean;
  claims: ClaimVerdict[];
  overall_risk_score: number;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function verifyResponse(
  prompt: string,
  response: string,
  runSemanticEntropy = true
): Promise<VerifyResponse> {
  const res = await fetch(`${API_BASE}/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, response, run_semantic_entropy: runSemanticEntropy }),
  });
  if (!res.ok) {
    throw new Error(`Verify request failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}
