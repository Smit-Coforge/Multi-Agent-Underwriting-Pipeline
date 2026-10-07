from __future__ import annotations

from typing import Any

import pytest

from src import tracing
from src.llm import FakeLLM

ALEX: dict[str, Any] = {
    "name": "Alex Rivera",
    "age": 34,
    "product": "term_life",
    "coverage": 500000,
    "smoker": False,
    "income": 85000,
    "claims": [{"year": 2021, "type": "auto", "severity": "minor"}],
}

JORDAN: dict[str, Any] = {
    "name": "Jordan Blake",
    "age": 52,
    "product": "term_life",
    "coverage": 1000000,
    "smoker": True,
    "income": 60000,
    "claims": [
        {"year": 2019, "type": "auto", "severity": "moderate"},
        {"year": 2022, "type": "home", "severity": "moderate"},
        {"year": 2023, "type": "auto", "severity": "major"},
    ],
}

CASEY: dict[str, Any] = {
    "name": "Casey Morgan",
    "age": 41,
    "product": "term_life",
    "coverage": 300000,
    "smoker": False,
    "income": 70000,
    "claims": [],
}

ALEX_RAW = {
    "text": (
        "Hi, I'm Alex Rivera, 34. I'd like $500k term life. "
        "I'm a non-smoker, I make about 85k. I had one small auto claim in 2021."
    )
}

JORDAN_RAW = {
    "text": (
        "I'm Jordan Blake, 52 years old. I want $1 million in term life. I smoke. "
        "Income is about $60k. Claims: 2019 auto, 2022 home water damage, "
        "and a 2023 at-fault auto accident."
    )
}

CASEY_RAW = {
    "text": (
        "Hi, Casey Morgan here, 41. Looking for $300k term life. "
        "Non-smoker, income around $70k, no prior claims."
    )
}


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
