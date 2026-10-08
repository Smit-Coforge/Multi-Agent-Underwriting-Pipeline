from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ENRICH_DIR = Path(__file__).resolve().parents[2] / "data" / "enrichment"

FIELDS = (
    "prescription_history",
    "recent_lab_results",
    "verified_medical_history",
    "financial_coverage_check",
)


def lookup_enrichment(case_id: str, root: Path = ENRICH_DIR) -> dict[str, Any]:
    path = root / f"{case_id}.json"
    if not path.is_file():
        raise ValueError(f"no enrichment file for {case_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in FIELDS if key not in data]
    if missing:
        raise ValueError(f"enrichment missing keys: {missing}")
    return {key: data[key] for key in FIELDS}
