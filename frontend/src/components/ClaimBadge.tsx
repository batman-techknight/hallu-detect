import type { ClaimVerdict } from "../api/client";

const COLORS: Record<ClaimVerdict["label"], string> = {
  supported: "#1e8e3e",
  contradicted: "#d93025",
  unverifiable: "#f9ab00",
};

export function ClaimBadge({ verdict }: { verdict: ClaimVerdict }) {
  return (
    <div style={{ borderLeft: `4px solid ${COLORS[verdict.label]}`, padding: "8px 12px", marginBottom: 8 }}>
      <strong>{verdict.claim}</strong>
      <div style={{ fontSize: 13, opacity: 0.8 }}>
        {verdict.label} · confidence {(verdict.confidence * 100).toFixed(0)}%
        {verdict.from_cache ? " · cached" : ""}
      </div>
      <div style={{ fontSize: 13 }}>{verdict.rationale}</div>
    </div>
  );
}
