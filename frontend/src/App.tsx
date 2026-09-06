import { useState } from "react";
import { verifyResponse, type VerifyResponse } from "./api/client";
import { ClaimBadge } from "./components/ClaimBadge";

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");
  const [result, setResult] = useState<VerifyResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      setResult(await verifyResponse(prompt, response));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 720, margin: "40px auto", fontFamily: "system-ui, sans-serif" }}>
      <h1>Hallucination Guard</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Prompt
          <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={2} style={{ width: "100%" }} />
        </label>
        <label>
          LLM response to check
          <textarea value={response} onChange={(e) => setResponse(e.target.value)} rows={4} style={{ width: "100%" }} />
        </label>
        <button type="submit" disabled={loading || !prompt || !response}>
          {loading ? "Checking…" : "Check for hallucinations"}
        </button>
      </form>

      {error && <p style={{ color: "#d93025" }}>{error}</p>}

      {result && (
        <div style={{ marginTop: 24 }}>
          <h2>Risk score: {(result.overall_risk_score * 100).toFixed(0)}%</h2>
          {result.semantic_entropy !== null && (
            <p>Semantic entropy: {result.semantic_entropy.toFixed(2)} {result.entropy_flagged ? "(flagged)" : ""}</p>
          )}
          {result.claims.map((c, i) => (
            <ClaimBadge key={i} verdict={c} />
          ))}
        </div>
      )}
    </div>
  );
}
