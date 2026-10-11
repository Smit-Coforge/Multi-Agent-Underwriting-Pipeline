import pytest

from src.agents.recommendation import pick_outcome, recommendation_agent
from src.llm import FakeLLM
from src.tools.enrichment_lookup import lookup_enrichment
from tests.conftest import ALEX

ENRICHMENT = lookup_enrichment("C-001")
RISK = {"score": 0.38, "band": "low", "factors": ["clean record"]}


def test_pick_outcome_thresholds() -> None:
    assert pick_outcome(0.38) == "approve"
    assert pick_outcome(0.4) == "refer"
    assert pick_outcome(0.7) == "refer"
    assert pick_outcome(0.81) == "deny"


def test_recommendation_keeps_valid_outcome_and_reason() -> None:
    llm = FakeLLM(
        {
            "recommendation": {
                "outcome": "approve",
                "reason": "Low risk file.",
                "rate_class": "standard",
            }
        }
    )
    out = recommendation_agent(ALEX, ENRICHMENT, RISK, case_id="C-001", llm=llm)
    assert out["outcome"] == "approve"
    assert out["reason"] == "Low risk file."
    assert out["rate_class"] == "standard"
    assert "review_checklist" not in out


def test_recommendation_stores_an_outcome_the_score_alone_would_not_pick() -> None:
    llm = FakeLLM(
        {
            "recommendation": {
                "outcome": "deny",
                "reason": "The file as a whole is not acceptable.",
            }
        }
    )
    out = recommendation_agent(ALEX, ENRICHMENT, RISK, case_id="C-001", llm=llm)
    assert out["outcome"] == "deny"
    assert out["reason"] == "The file as a whole is not acceptable."
    assert "review_checklist" not in out


def test_approve_rate_class_must_match_the_assessment() -> None:
    priced = {**ENRICHMENT, "rate_assessment": {"surcharge": True, "reason": "DUI on the MVR."}}
    standard = FakeLLM(
        {
            "recommendation": {
                "outcome": "approve",
                "reason": "Issue it.",
                "rate_class": "standard",
            }
        }
    )
    with pytest.raises(ValueError):
        recommendation_agent(ALEX, priced, RISK, case_id="C-002", llm=standard)
    surcharge = FakeLLM(
        {
            "recommendation": {
                "outcome": "approve",
                "reason": "Issue at a higher price.",
                "rate_class": "surcharge",
            }
        }
    )
    out = recommendation_agent(ALEX, priced, RISK, case_id="C-002", llm=surcharge)
    assert out["rate_class"] == "surcharge"


def test_recommendation_rejects_unknown_outcome() -> None:
    llm = FakeLLM({"recommendation": {"outcome": "postpone", "reason": "Wait."}})
    with pytest.raises(ValueError):
        recommendation_agent(ALEX, ENRICHMENT, RISK, case_id="C-001", llm=llm)


def test_recommendation_raises_when_llm_fails() -> None:
    llm = FakeLLM({"recommendation": "not-json"})
    with pytest.raises(ValueError):
        recommendation_agent(ALEX, ENRICHMENT, RISK, case_id="C-001", llm=llm)


def test_recommendation_raises_on_empty_reason() -> None:
    llm = FakeLLM(
        {"recommendation": {"outcome": "approve", "reason": "  ", "rate_class": "standard"}}
    )
    with pytest.raises(ValueError):
        recommendation_agent(ALEX, ENRICHMENT, RISK, case_id="C-001", llm=llm)


def test_refer_includes_review_checklist() -> None:
    llm = FakeLLM(
        {
            "recommendation": {
                "outcome": "refer",
                "reason": "Refer for a human review of the speeding ticket.",
                "review_checklist": ["Confirm the speeding ticket", "Review mileage"],
            }
        }
    )
    risk = {"score": 0.55, "band": "medium", "factors": ["One speeding ticket"]}
    out = recommendation_agent(ALEX, ENRICHMENT, risk, case_id="C-004", llm=llm)
    assert out["outcome"] == "refer"
    assert out["reason"] == "Refer for a human review of the speeding ticket."
    assert out["review_checklist"] == ["Confirm the speeding ticket", "Review mileage"]


def test_refer_raises_when_llm_fails() -> None:
    llm = FakeLLM({"recommendation": "not-json"})
    risk = {"score": 0.55, "band": "medium", "factors": ["One speeding ticket"]}
    with pytest.raises(ValueError):
        recommendation_agent(ALEX, ENRICHMENT, risk, case_id="C-004", llm=llm)


def test_refer_raises_on_bad_checklist() -> None:
    llm = FakeLLM(
        {
            "recommendation": {
                "outcome": "refer",
                "reason": "Refer for a human review.",
                "review_checklist": [],
            }
        }
    )
    risk = {"score": 0.55, "band": "medium", "factors": ["One speeding ticket"]}
    with pytest.raises(ValueError):
        recommendation_agent(ALEX, ENRICHMENT, risk, case_id="C-004", llm=llm)


class _TimeoutLLM(FakeLLM):
    def complete(self, messages: list[dict[str, str]], *, agent: str) -> str:
        raise TimeoutError("simulated LLM timeout")


def test_recommendation_timeout_raises() -> None:
    llm = _TimeoutLLM({})
    with pytest.raises(TimeoutError):
        recommendation_agent(ALEX, ENRICHMENT, RISK, case_id="C-001", llm=llm)
