from __future__ import annotations

import json
from typing import Any

from src.llm import parse_json_object
from src.state import ALLOWED_CONDITIONS

REQUIRED = (
    "name",
    "age",
    "gender",
    "occupation",
    "annual_income",
    "coverage_amount",
    "term_length_years",
    "smoker",
    "medical_conditions",
    "existing_life_insurance",
)

APPLICATION_FIELDS = REQUIRED + ("medical_conditions_text",)

_LABELS = ", ".join(ALLOWED_CONDITIONS)
_FIELDS = ", ".join(APPLICATION_FIELDS)

INTAKE_SYSTEM = f"""You standardize a term life application. Use only the user JSON.
Return JSON only, with exactly two keys and no others:
medical_conditions (array of strings) and consistency_flags (array of objects).
Do not return name, age, gender, occupation, income, coverage, term, smoker, or existing insurance. Code copies those from the form.

medical_conditions:
Map every condition disclosed in medical_conditions and medical_conditions_text onto this allowed list and no other labels:
{_LABELS}
If nothing is disclosed, return an empty array.
If a disclosed value is already one of those labels, keep that label. Do not map prediabetes to type2_diabetes.
Also map these phrases:
- high BP or high blood pressure -> hypertension
- borderline sugar -> prediabetes
- diabetes or type 2 diabetes -> type2_diabetes
- high cholesterol or hyperlipidemia -> high_cholesterol
- anxiety or depression -> depression_anxiety
- coronary artery disease or CAD -> cad
- COPD -> copd
- cancer -> cancer_history
- stroke -> stroke_history
Use other only when a condition is disclosed and none of the labels above fit.
Do not invent a condition the form does not mention.

consistency_flags:
Each object has a fields array and an issue string.
Every name in fields must be one of:
{_FIELDS}
The checks below are separate. Add one flag for each check that is true. Do not require the others to be true as well.
1. Occupation is a student, intern, or unemployed, and annual_income is very high. fields: occupation, annual_income.
2. age plus term_length_years is greater than 80. fields: age, term_length_years.
3. smoker is false and COPD is disclosed (the word COPD or the label copd). fields: smoker, medical_conditions.
4. coverage_amount divided by annual_income is 15 or higher. fields: annual_income, coverage_amount.
If none of the checks are true, consistency_flags is an empty array."""


def _form(raw_application: Any) -> dict[str, Any]:
    if not isinstance(raw_application, dict):
        raise ValueError("application must be a JSON object")  # noqa: TRY004
    missing = [key for key in REQUIRED if key not in raw_application]
    if missing:
        raise ValueError(f"intake missing keys: {missing}")
    return raw_application


def _conditions(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError("medical_conditions must be a list of allowed labels")  # noqa: TRY004
    labels: list[str] = []
    for item in raw:
        if item not in ALLOWED_CONDITIONS:
            raise ValueError(f"condition not allowed: {item!r}")
        if item not in labels:
            labels.append(item)
    return labels


def _flags(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("consistency_flags must be a list")  # noqa: TRY004
    flags: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("consistency flag must be an object")  # noqa: TRY004
        fields = item.get("fields")
        issue = item.get("issue")
        if not isinstance(fields, list) or not fields:
            raise ValueError("consistency flag fields must be a non-empty list")
        cleaned: list[str] = []
        for field in fields:
            if field not in APPLICATION_FIELDS:
                raise ValueError(f"unknown application field: {field!r}")
            if field not in cleaned:
                cleaned.append(field)
        if not isinstance(issue, str) or not issue.strip():
            raise ValueError("consistency flag issue must be a non-empty string")
        flags.append({"fields": cleaned, "issue": issue.strip()})
    return flags


def intake_agent(raw_application: Any, *, case_id: str, llm: Any) -> dict[str, Any]:
    form = _form(raw_application)
    content = llm.complete(
        [
            {"role": "system", "content": INTAKE_SYSTEM},
            {"role": "user", "content": json.dumps(form)},
        ],
        agent="intake",
    )
    data = parse_json_object(content)
    applicant = {key: form[key] for key in REQUIRED}
    applicant["medical_conditions"] = _conditions(data.get("medical_conditions"))
    if "medical_conditions_text" in form:
        text = form["medical_conditions_text"]
        if not isinstance(text, str):
            raise ValueError("medical_conditions_text must be a string")
        applicant["medical_conditions_text"] = text
    applicant["consistency_flags"] = _flags(data.get("consistency_flags"))
    return applicant
