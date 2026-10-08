from __future__ import annotations

import json
from typing import Any

from src.tools.enrichment_lookup import lookup_enrichment


def enrichment_agent(applicant: dict[str, Any], *, case_id: str, llm: Any) -> dict[str, Any]:
    looked_up = lookup_enrichment(case_id)
    facts = {"applicant": applicant, **looked_up}
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
    return {**looked_up, "summary": data["summary"]}
