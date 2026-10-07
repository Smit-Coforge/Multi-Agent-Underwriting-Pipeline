from __future__ import annotations

import json
from typing import Any

from src.mocks.claims_history import prior_claims_lookup
from src.mocks.credit_bureau import credit_band_lookup

HIGH_VALUE = 500_000


def enrichment_agent(applicant: dict[str, Any], *, case_id: str, llm: Any) -> dict[str, Any]:
    credit_band = credit_band_lookup(int(applicant.get("income") or 0))
    prior_claims = prior_claims_lookup(applicant.get("claims") or [])
    high_value_flag = int(applicant.get("coverage") or 0) >= HIGH_VALUE
    facts = {
        "applicant": applicant,
        "credit_band": credit_band,
        "prior_claims": prior_claims,
        "high_value_flag": high_value_flag,
    }
    content = llm.complete(
        [
            {
                "role": "system",
                "content": (
                    "Write one sentence summarizing this term life file for an "
                    'underwriter. Return JSON only: {"summary": "..."}. '
                    "Use only the user JSON."
                ),
            },
            {"role": "user", "content": json.dumps(facts)},
        ],
        agent="enrichment",
    )
    data = json.loads(content)
    return {
        "credit_band": credit_band,
        "prior_claims": prior_claims,
        "high_value_flag": high_value_flag,
        "summary": data["summary"],
    }
