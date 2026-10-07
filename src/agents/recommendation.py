from __future__ import annotations

import json
from typing import Any

REC_SYSTEM = (
    "You are a term life underwriter. The approve/refer/deny outcome is already "
    "decided from the risk score and must not be changed. Write one sentence "
    "explaining that outcome from the file. Return JSON only: {\"reason\": \"...\"}. "
    "Use only the user JSON. Do not invent medical facts."
)


def pick_outcome(score: float) -> str:
    if score < 0.4:
        return "approve"
    if score <= 0.7:
        return "refer"
    return "deny"


def recommendation_agent(
    applicant: dict[str, Any],
    enrichment: dict[str, Any],
    risk: dict[str, Any],
    *,
    case_id: str,
    llm: Any,
) -> dict[str, Any]:
    score = float(risk["score"])
    outcome = pick_outcome(score)
    try:
        content = llm.complete(
            [
                {"role": "system", "content": REC_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "outcome": outcome,
                            "score": score,
                            "applicant": applicant,
                            "enrichment": enrichment,
                            "risk": risk,
                        }
                    ),
                },
            ],
            agent="recommendation",
        )
        reason = json.loads(content)["reason"]
    except Exception:  # noqa: BLE001 — keep code-picked outcome if the LLM reason fails
        reason = f"Score {score} mapped to {outcome} by threshold."
    return {"outcome": outcome, "reason": reason}
