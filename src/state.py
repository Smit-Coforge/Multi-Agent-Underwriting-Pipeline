from typing import Any, TypedDict

# Labels the intake and enrichment models may propose. Code rejects anything else.
ALLOWED_CONDITIONS: tuple[str, ...] = (
    "hypertension",
    "prediabetes",
    "type2_diabetes",
    "high_cholesterol",
    "asthma",
    "depression_anxiety",
    "sleep_apnea",
    "cad",
    "copd",
    "cancer_history",
    "stroke_history",
    "other",
)

CONTRACTS: dict[str, dict[str, Any]] = {
    "intake": {"reads": ["raw"], "writes": "applicant"},
    "enrichment": {"reads": ["applicant"], "writes": "enrichment"},
    "risk": {"reads": ["applicant", "enrichment"], "writes": "risk"},
    "recommendation": {
        "reads": ["applicant", "enrichment", "risk"],
        "writes": "decision",
    },
}


class CaseState(TypedDict):
    case_id: str
    raw: Any
    applicant: dict[str, Any] | None
    enrichment: dict[str, Any] | None
    risk: dict[str, Any] | None
    decision: dict[str, Any] | None
    meta: dict[str, Any]


def new_case(case_id: str, raw: Any) -> CaseState:
    return {
        "case_id": case_id,
        "raw": raw,
        "applicant": None,
        "enrichment": None,
        "risk": None,
        "decision": None,
        "meta": {"errors": [], "retries": {}},
    }
