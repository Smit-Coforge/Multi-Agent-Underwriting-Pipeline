from __future__ import annotations

import json
from typing import Any

RISK_SYSTEM = (
    "You are a term life underwriter. A numeric risk score on [0, 1] is already "
    "computed (0 = preferred, 1 = decline). List the mortality and financial "
    "factors that apply. Return JSON only: {\"factors\": [\"...\"]}. "
    "Use only the user JSON. Do not invent labs, diagnoses, or a different score."
)


def _at_fault(claims: list[Any]) -> bool:
    for claim in claims:
        blob = json.dumps(claim).lower() if not isinstance(claim, str) else claim.lower()
        if "at-fault" in blob or "at fault" in blob:
            return True
    return False


def compute_term_life_score(
    applicant: dict[str, Any],
    enrichment: dict[str, Any],
) -> tuple[float, str, list[str]]:
    score = 0.10
    factors: list[str] = []

    if applicant.get("smoker"):
        score += 0.35
        factors.append("Current tobacco use")

    age = int(applicant.get("age") or 0)
    if age >= 50:
        score += 0.18
        factors.append(f"Age {age} (50+ mortality class)")
    elif age >= 40:
        score += 0.08
        factors.append(f"Age {age}")

    credit = enrichment.get("credit_band") or "fair"
    if credit == "poor":
        score += 0.12
        factors.append("Poor credit band")
    elif credit == "fair":
        score += 0.06
        factors.append("Fair credit band")

    claims = list(applicant.get("claims") or [])
    prior = int(enrichment.get("prior_claims") or len(claims))
    if prior:
        score += 0.06 * prior
        factors.append(f"{prior} prior claim(s) on record")
    if _at_fault(claims):
        score += 0.10
        factors.append("At-fault driving incident")

    income = int(applicant.get("income") or 0)
    coverage = int(applicant.get("coverage") or 0)
    if income > 0:
        ratio = coverage / income
        if ratio > 15:
            score += 0.20
            factors.append(f"Face amount {ratio:.0f}x income (financial over-insurance)")
        elif ratio > 10:
            score += 0.08
            factors.append(f"Face amount {ratio:.0f}x income")

    score = min(1.0, max(0.0, round(score, 2)))
    if score < 0.4:
        band = "low"
    elif score <= 0.7:
        band = "medium"
    else:
        band = "high"
    if not factors:
        factors.append("No extra mortality or financial debits")
    return score, band, factors


def risk_scoring_agent(
    applicant: dict[str, Any],
    enrichment: dict[str, Any],
    *,
    case_id: str,
    llm: Any,
) -> dict[str, Any]:
    score, band, computed = compute_term_life_score(applicant, enrichment)
    content = llm.complete(
        [
            {"role": "system", "content": RISK_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "applicant": applicant,
                        "enrichment": enrichment,
                        "score": score,
                        "band": band,
                    }
                ),
            },
        ],
        agent="risk",
    )
    data = json.loads(content)
    factors = data.get("factors")
    if not isinstance(factors, list) or not factors:
        factors = computed
    return {"score": score, "band": band, "factors": factors}
