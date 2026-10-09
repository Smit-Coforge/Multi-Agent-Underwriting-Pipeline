from __future__ import annotations

from typing import Any

import pytest

from src import tracing
from src.llm import FakeLLM

ALEX_RAW: dict[str, Any] = {
    "name": "Alex Rivera",
    "age": 34,
    "gender": "female",
    "occupation": "software engineer",
    "annual_income": 85000,
    "coverage_amount": 500000,
    "term_length_years": 20,
    "smoker": False,
    "medical_conditions": [],
    "existing_life_insurance": 0,
}

ALEX: dict[str, Any] = {**ALEX_RAW, "consistency_flags": []}

JORDAN_RAW: dict[str, Any] = {
    "name": "Jordan Blake",
    "age": 54,
    "gender": "male",
    "occupation": "construction supervisor",
    "annual_income": 60000,
    "coverage_amount": 1000000,
    "term_length_years": 20,
    "smoker": True,
    "medical_conditions": ["COPD", "hypertension"],
    "existing_life_insurance": 100000,
}

JORDAN: dict[str, Any] = {
    **JORDAN_RAW,
    "medical_conditions": ["copd", "hypertension"],
    "consistency_flags": [],
}

CASEY_RAW: dict[str, Any] = {
    "name": "Casey Morgan",
    "age": 41,
    "gender": "nonbinary",
    "occupation": "teacher",
    "annual_income": 70000,
    "coverage_amount": 300000,
    "term_length_years": 20,
    "smoker": False,
    "medical_conditions": [],
    "existing_life_insurance": 0,
}

CASEY: dict[str, Any] = {**CASEY_RAW, "consistency_flags": []}


def fake_for(applicant: dict[str, Any]) -> FakeLLM:
    return FakeLLM(
        {
            "intake": {
                "medical_conditions": applicant.get("medical_conditions", []),
                "consistency_flags": applicant.get("consistency_flags", []),
            },
            "enrichment": {
                "nondisclosure_findings": [],
                "coverage_assessment": {
                    "justified": True,
                    "reason": "Coverage is consistent with income.",
                },
            },
            "risk": {
                "score": 0.99,
                "factors": ["not the official factors"],
                "underwriter_note": "Note from the model.",
            },
            "recommendation": {
                "reason": "Threshold applied to the risk score.",
                "review_checklist": ["Review the file before issuing."],
            },
        }
    )


class TimeoutOnRiskLLM(FakeLLM):
    def complete(self, messages: list[dict[str, str]], *, agent: str) -> str:
        if agent == "risk":
            raise TimeoutError("simulated LLM timeout")
        return super().complete(messages, agent=agent)


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracing.time, "sleep", lambda _seconds: None)
