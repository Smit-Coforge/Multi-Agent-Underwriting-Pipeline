from __future__ import annotations

import json
from typing import Any

from src.llm import parse_json_object


def _system(outcome: str) -> str:
    if outcome == "refer":
        shape = '{"reason": "...", "review_checklist": ["short item"]}'
        extra = (
            " review_checklist is a short list of concrete items for the human "
            "underwriter, drawn from those flags and findings."
        )
    else:
        shape = '{"reason": "..."}'
        extra = ""
    return (
        "You are a term life underwriter. The outcome is already decided and "
        f"is {outcome}. Explain that outcome in one sentence using "
        "consistency_flags, nondisclosure_findings, coverage_assessment, and "
        "factors from the user JSON. Do not invent medical facts and do not "
        f"choose a different outcome. Return JSON only: {shape}.{extra}"
    )


def pick_outcome(score: float) -> str:
    if score < 0.4:
        return "approve"
    if score <= 0.7:
        return "refer"
    return "deny"


def _checklist(raw: Any) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("review_checklist must be a non-empty list")
    items: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("review_checklist items must be non-empty strings")
        items.append(item.strip())
    return items


def _fallback_checklist(
    applicant: dict[str, Any],
    enrichment: dict[str, Any],
    risk: dict[str, Any],
) -> list[str]:
    items: list[str] = []
    for flag in applicant.get("consistency_flags") or []:
        if isinstance(flag, dict):
            issue = flag.get("issue")
            if isinstance(issue, str) and issue.strip():
                items.append(issue.strip())
    for finding in enrichment.get("nondisclosure_findings") or []:
        if isinstance(finding, dict) and isinstance(finding.get("evidence"), str):
            suggests = finding.get("suggests") or "a condition"
            items.append(f"Review {finding['evidence']} (suggests {suggests})")
    coverage = enrichment.get("coverage_assessment") or {}
    if isinstance(coverage, dict) and coverage.get("justified") is False:
        reason = coverage.get("reason")
        if isinstance(reason, str) and reason.strip():
            items.append(reason.strip())
    if not items:
        for factor in risk.get("factors") or []:
            if isinstance(factor, str) and factor.strip():
                items.append(factor.strip())
    return items[:5] or ["Review the application against the file."]


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
    fallback = f"Score {score} mapped to {outcome} by threshold."
    reason = fallback
    checklist: list[str] | None = None
    try:
        content = llm.complete(
            [
                {"role": "system", "content": _system(outcome)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "outcome": outcome,
                            "score": score,
                            "factors": risk.get("factors") or [],
                            "consistency_flags": applicant.get("consistency_flags") or [],
                            "nondisclosure_findings": enrichment.get("nondisclosure_findings")
                            or [],
                            "coverage_assessment": enrichment.get("coverage_assessment") or {},
                            "applicant": {
                                "name": applicant.get("name"),
                                "age": applicant.get("age"),
                                "occupation": applicant.get("occupation"),
                                "smoker": applicant.get("smoker"),
                                "medical_conditions": applicant.get("medical_conditions") or [],
                                "annual_income": applicant.get("annual_income"),
                                "coverage_amount": applicant.get("coverage_amount"),
                                "term_length_years": applicant.get("term_length_years"),
                            },
                        }
                    ),
                },
            ],
            agent="recommendation",
        )
        data = parse_json_object(content)
        proposed = data.get("reason")
        if not isinstance(proposed, str) or not proposed.strip():
            raise ValueError("reason must be a non-empty string")
        reason = proposed.strip()
        if outcome == "refer":
            checklist = _checklist(data.get("review_checklist"))
    except Exception:  # noqa: BLE001 — keep code-picked outcome if the LLM reason fails
        reason = fallback
        checklist = None
    decision: dict[str, Any] = {"outcome": outcome, "reason": reason}
    if outcome == "refer":
        decision["review_checklist"] = checklist or _fallback_checklist(
            applicant, enrichment, risk
        )
    return decision
