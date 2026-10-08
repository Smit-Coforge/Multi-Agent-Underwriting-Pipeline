from __future__ import annotations

import json
from typing import Any

RISK_SYSTEM = (
    "You are a term life underwriter. A numeric risk score on [0, 1] is already "
    "computed (0 = preferred, 1 = decline). List the mortality and financial "
    "factors that apply. Return JSON only: {\"factors\": [\"...\"]}. "
    "Use only the user JSON. Do not invent labs, diagnoses, or a different score."
)

SEVERE = ("cad", "copd", "cancer", "stroke")


def _conditions(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def compute_term_life_score(
    applicant: dict[str, Any],
    enrichment: dict[str, Any],
) -> tuple[float, str, list[str]]:
    score = 0.10
    factors: list[str] = []

    if applicant.get("smoker"):
        score += 0.30
        factors.append("Current tobacco use")

    age = int(applicant.get("age") or 0)
    if age >= 55:
        score += 0.16
        factors.append(f"Age {age} (55+ mortality class)")
    elif age >= 45:
        score += 0.08
        factors.append(f"Age {age}")

    disclosed = _conditions(applicant.get("medical_conditions"))
    if disclosed:
        score += min(0.08 * len(disclosed), 0.24)
        factors.append(f"{len(disclosed)} disclosed medical condition(s)")

    verified = enrichment.get("verified_medical_history") or {}
    verified_conditions = _conditions(verified.get("conditions"))
    history = " ".join(disclosed + verified_conditions).lower()
    if any(token in history for token in SEVERE):
        score += 0.15
        factors.append("Severe condition on file")

    if verified.get("matches_application") is False:
        score += 0.12
        factors.append("Verified history does not match the application")

    labs = enrichment.get("recent_lab_results") or {}
    flags = _conditions(labs.get("flags"))
    if flags:
        score += min(0.08 * len(flags), 0.16)
        factors.append(f"{len(flags)} abnormal lab flag(s)")

    prescriptions = enrichment.get("prescription_history") or []
    if isinstance(prescriptions, list) and prescriptions:
        score += min(0.04 * len(prescriptions), 0.12)
        factors.append(f"{len(prescriptions)} current prescription(s)")

    financial = enrichment.get("financial_coverage_check") or {}
    if financial.get("consistent") is False:
        score += 0.18
        factors.append("Coverage is inconsistent with income")

    income = int(applicant.get("annual_income") or 0)
    coverage = int(applicant.get("coverage_amount") or 0)
    if income > 0:
        ratio = coverage / income
        if ratio > 15:
            score += 0.20
            factors.append(f"Face amount {ratio:.0f}x income")
        elif ratio > 10:
            score += 0.10
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
