from __future__ import annotations

import json
from typing import Any

from src.llm import parse_json_object

RECOMMENDATION_SYSTEM = (
    "You are an auto insurance underwriter. Choose the outcome from the whole file, "
    "not from a score cutoff. approve when the driver and car can be issued. "
    "refer when a person should look at something specific. deny when the file "
    "should not be issued. A bad record raises the price or is a decline. "
    "A consistency flag is an example to consider, not a rule that forces refer or deny. "
    "On approve, include rate_class. Use surcharge when rate_assessment.surcharge "
    "is true, and standard when it is false. refer and deny do not include rate_class. "
    'Return JSON only. For approve: {"outcome": "approve", "reason": "...", '
    '"rate_class": "standard"}. For deny: {"outcome": "deny", "reason": "..."}. '
    "For refer, also include review_checklist, a short list of concrete items "
    'for the human underwriter: {"outcome": "refer", "reason": "...", '
    '"review_checklist": ["short item"]}. '
    "reason is one sentence. Use only the user JSON. Do not invent tickets or accidents."
)

_OUTCOMES = ("approve", "refer", "deny")
_RATE_CLASSES = ("standard", "surcharge")


def pick_outcome(score: float) -> str:
    if score < 0.4:
        return "approve"
    if score <= 0.7:
        return "refer"
    return "deny"


def _expected_rate_class(enrichment: dict[str, Any]) -> str:
    assessment = enrichment.get("rate_assessment") or {}
    if isinstance(assessment, dict) and assessment.get("surcharge") is True:
        return "surcharge"
    return "standard"


def _checklist(raw: Any) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("review_checklist must be a non-empty list")
    items: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("review_checklist items must be non-empty strings")
        items.append(item.strip())
    return items


def recommendation_agent(
    applicant: dict[str, Any],
    enrichment: dict[str, Any],
    risk: dict[str, Any],
    *,
    case_id: str,
    llm: Any,
) -> dict[str, Any]:
    driver = {
        "name": applicant.get("name"),
        "age": applicant.get("age"),
        "zip_code": applicant.get("zip_code"),
        "years_licensed": applicant.get("years_licensed"),
        "vehicle_year": applicant.get("vehicle_year"),
        "vehicle_make": applicant.get("vehicle_make"),
        "vehicle_model": applicant.get("vehicle_model"),
        "annual_mileage": applicant.get("annual_mileage"),
        "coverage_level": applicant.get("coverage_level"),
        "violations": applicant.get("violations") or [],
        "accidents_disclosed": applicant.get("accidents_disclosed"),
    }
    if applicant.get("driving_record_text"):
        driver["driving_record_text"] = applicant.get("driving_record_text")
    content = llm.complete(
        [
            {"role": "system", "content": RECOMMENDATION_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "risk": risk,
                        "consistency_flags": applicant.get("consistency_flags") or [],
                        "nondisclosure_findings": enrichment.get("nondisclosure_findings")
                        or [],
                        "rate_assessment": enrichment.get("rate_assessment") or {},
                        "driver": driver,
                    }
                ),
            },
        ],
        agent="recommendation",
    )
    data = parse_json_object(content)
    outcome = data.get("outcome")
    if outcome not in _OUTCOMES:
        raise ValueError("outcome must be approve, refer, or deny")
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    decision: dict[str, Any] = {"outcome": outcome, "reason": reason.strip()}
    if outcome == "approve":
        rate_class = data.get("rate_class")
        if rate_class not in _RATE_CLASSES:
            raise ValueError("rate_class must be standard or surcharge")
        if rate_class != _expected_rate_class(enrichment):
            raise ValueError("rate_class does not match the rate assessment")
        decision["rate_class"] = rate_class
    if outcome == "refer":
        decision["review_checklist"] = _checklist(data.get("review_checklist"))
    return decision
