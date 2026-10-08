from __future__ import annotations

import json
from typing import Any

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


def intake_agent(raw_application: Any, *, case_id: str, llm: Any) -> dict[str, Any]:
    text = (
        raw_application
        if isinstance(raw_application, str)
        else raw_application.get("text", json.dumps(raw_application))
    )
    content = llm.complete(
        [
            {
                "role": "system",
                "content": (
                    "Extract a term life application into JSON with keys: "
                    "name (string), age (integer years), gender (string), "
                    "occupation (string), annual_income (integer USD), "
                    "coverage_amount (integer USD face amount), "
                    "term_length_years (integer), smoker (boolean), "
                    "medical_conditions (list of strings), "
                    "existing_life_insurance (integer USD already in force, 0 if none). "
                    "Use only the user message. Do not invent medical history."
                ),
            },
            {"role": "user", "content": text},
        ],
        agent="intake",
    )
    data = json.loads(content)
    missing = [key for key in REQUIRED if key not in data]
    if missing:
        raise ValueError(f"intake missing keys: {missing}")
    return data
