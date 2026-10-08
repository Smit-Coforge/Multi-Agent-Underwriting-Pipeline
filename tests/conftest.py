from __future__ import annotations

from typing import Any

import pytest

from src import tracing
from src.llm import FakeLLM

ALEX: dict[str, Any] = {
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

JORDAN: dict[str, Any] = {
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

CASEY: dict[str, Any] = {
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

ALEX_RAW = ALEX
JORDAN_RAW = JORDAN
CASEY_RAW = CASEY


def fake_for(applicant: dict[str, Any]) -> FakeLLM:
    return FakeLLM(
        {
            "intake": applicant,
            "enrichment": {"summary": "Term life file summary."},
            "risk": {"factors": ["computed from file"]},
            "recommendation": {"reason": "Threshold applied to the risk score."},
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
