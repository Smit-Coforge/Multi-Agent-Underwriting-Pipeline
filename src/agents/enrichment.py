from __future__ import annotations

import json
from typing import Any

from src.llm import parse_json_object
from src.state import ALLOWED_CONDITIONS
from src.tools.enrichment_lookup import lookup_enrichment

_LABELS = ", ".join(ALLOWED_CONDITIONS)
_EVIDENCE_TYPES = ("prescription", "lab", "medical_history")

ENRICH_SYSTEM = f"""You compare outside records to what the applicant already disclosed. Use only the user JSON.
Return JSON only, with exactly two keys and no others:
nondisclosure_findings (array of objects) and coverage_assessment (object).
Do not return the prescription, lab, or history records. Code copies those from the file.
disclosed_conditions is already a list of standardized labels. Do not remap the application.

nondisclosure_findings:
A finding is a condition the records show that is not in disclosed_conditions.
Each object has evidence_type, evidence, suggests, and explanation.
evidence_type is one of: prescription, lab, medical_history.
evidence must be copied exactly from the records. Do not paraphrase it and do not return a number.
- prescription: the drug string from prescription_history.
- lab: the key only, one of a1c, systolic, diastolic, or ldl. Write a1c, not "a1c: 5.2". Not date and not flags.
- medical_history: one string from verified_medical_history.conditions, copied as written.
suggests is the one label that record points to. It must be one of:
{_LABELS}
Use that label, not the wording on the record. If suggests is already in disclosed_conditions, do not emit that finding.
explanation is one sentence tying that evidence to the undisclosed condition.
If every condition in the records is already disclosed, or the records suggest nothing, nondisclosure_findings is an empty array.

coverage_assessment:
This is about the requested face amount, not about medical conditions.
justified is a JSON boolean, true or false, not a string. reason is one sentence.
Use requested_coverage_to_income. Do not recompute it. A ratio of 20 is 15 or higher.
The checks below are separate. Either one is enough to set justified to false.
1. requested_coverage_to_income is 15 or higher.
2. The ratio is under 15, but the face amount does not fit this age and term.
Set justified to true only when the ratio is under 15 and the amount fits this age and term."""


_LAB_KEYS = ("a1c", "systolic", "diastolic", "ldl")


def _lab_key(evidence: str, labs: dict[str, Any]) -> str | None:
    if evidence in _LAB_KEYS and evidence in labs:
        return evidence
    head = evidence.split(":", 1)[0].strip().lower()
    if head in _LAB_KEYS and head in labs:
        return head
    return None


def _lab_supports(labs: dict[str, Any], key: str, suggests: str) -> bool:
    value = labs.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if key == "a1c" and suggests == "type2_diabetes":
        return value >= 6.5
    if key == "a1c" and suggests == "prediabetes":
        return 5.7 <= value < 6.5
    if key == "systolic" and suggests == "hypertension":
        return value >= 140
    if key == "diastolic" and suggests == "hypertension":
        return value >= 90
    if key == "ldl" and suggests == "high_cholesterol":
        return value >= 160
    return False


def _evidence(looked_up: dict[str, Any]) -> dict[str, set[str]]:
    drugs: set[str] = set()
    for item in looked_up.get("prescription_history") or []:
        if isinstance(item, dict) and isinstance(item.get("drug"), str):
            drugs.add(item["drug"])
    labs = looked_up.get("recent_lab_results") or {}
    lab_keys = {str(key) for key in labs} if isinstance(labs, dict) else set()
    history = looked_up.get("verified_medical_history") or {}
    conditions: set[str] = set()
    if isinstance(history, dict):
        for item in history.get("conditions") or []:
            if isinstance(item, str):
                conditions.add(item)
    return {
        "prescription": drugs,
        "lab": lab_keys,
        "medical_history": conditions,
    }


def _findings(
    raw: Any,
    looked_up: dict[str, Any],
    disclosed: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("nondisclosure_findings must be a list")  # noqa: TRY004
    known = _evidence(looked_up)
    findings: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("nondisclosure finding must be an object")  # noqa: TRY004
        evidence_type = item.get("evidence_type")
        evidence = item.get("evidence")
        suggests = item.get("suggests")
        explanation = item.get("explanation")
        if evidence_type not in _EVIDENCE_TYPES:
            raise ValueError(f"invalid evidence_type: {evidence_type!r}")
        if not isinstance(evidence, str):
            raise ValueError(  # noqa: TRY004
                f"evidence not in {evidence_type} records: {evidence!r}"
            )
        if evidence_type == "lab":
            labs = looked_up.get("recent_lab_results") or {}
            if not isinstance(labs, dict):
                labs = {}
            key = _lab_key(evidence, labs)
            if key is None:
                raise ValueError(f"evidence not in lab records: {evidence!r}")
            if not isinstance(suggests, str) or not _lab_supports(labs, key, suggests):
                continue
            evidence = key
        elif evidence not in known[evidence_type]:
            raise ValueError(
                f"evidence not in {evidence_type} records: {evidence!r}"
            )
        if suggests not in ALLOWED_CONDITIONS:
            raise ValueError(f"suggests not allowed: {suggests!r}")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("finding explanation must be a non-empty string")
        if suggests in disclosed:
            continue
        findings.append(
            {
                "evidence_type": evidence_type,
                "evidence": evidence,
                "suggests": suggests,
                "explanation": explanation.strip(),
            }
        )
    return findings


def _coverage(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("coverage_assessment must be an object")  # noqa: TRY004
    justified = raw.get("justified")
    reason = raw.get("reason")
    if not isinstance(justified, bool):
        raise ValueError("coverage_assessment.justified must be a bool")  # noqa: TRY004
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("coverage_assessment.reason must be a non-empty string")
    return {"justified": justified, "reason": reason.strip()}


def enrichment_agent(applicant: dict[str, Any], *, case_id: str, llm: Any) -> dict[str, Any]:
    looked_up = lookup_enrichment(case_id)
    disclosed = [
        item
        for item in (applicant.get("medical_conditions") or [])
        if isinstance(item, str)
    ]
    income = int(applicant.get("annual_income") or 0)
    coverage = int(applicant.get("coverage_amount") or 0)
    payload = {
        "disclosed_conditions": disclosed,
        "requested_coverage_to_income": round(coverage / income, 2) if income else None,
        "medical_conditions_text": applicant.get("medical_conditions_text"),
        "age": applicant.get("age"),
        "occupation": applicant.get("occupation"),
        "annual_income": applicant.get("annual_income"),
        "coverage_amount": applicant.get("coverage_amount"),
        "term_length_years": applicant.get("term_length_years"),
        "existing_life_insurance": applicant.get("existing_life_insurance"),
        **looked_up,
    }
    content = llm.complete(
        [
            {"role": "system", "content": ENRICH_SYSTEM},
            {"role": "user", "content": json.dumps(payload)},
        ],
        agent="enrichment",
    )
    data = parse_json_object(content)
    return {
        **looked_up,
        "nondisclosure_findings": _findings(
            data.get("nondisclosure_findings"), looked_up, set(disclosed)
        ),
        "coverage_assessment": _coverage(data.get("coverage_assessment")),
    }
