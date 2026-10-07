from __future__ import annotations

import json
from typing import Any

REQUIRED = ("name", "age", "product", "coverage", "smoker", "income", "claims")


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
                    "name (string), age (integer years), product (always "
                    '"term_life"), coverage (integer USD face amount), '
                    "smoker (boolean), income (integer annual USD), "
                    "claims (list of {year, type, severity}). "
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
