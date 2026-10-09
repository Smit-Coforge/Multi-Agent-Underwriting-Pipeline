from src.agents.recommendation import pick_outcome, recommendation_agent
from src.llm import FakeLLM
from src.tools.enrichment_lookup import lookup_enrichment
from tests.conftest import ALEX

ENRICHMENT = lookup_enrichment("C-001")
RISK = {"score": 0.38, "band": "low", "factors": ["non-smoker"]}


def test_pick_outcome_thresholds() -> None:
    assert pick_outcome(0.38) == "approve"
    assert pick_outcome(0.4) == "refer"
    assert pick_outcome(0.7) == "refer"
    assert pick_outcome(0.81) == "deny"


def test_recommendation_agent_code_picks_outcome() -> None:
    llm = FakeLLM({"recommendation": {"outcome": "deny", "reason": "Low risk file."}})
    out = recommendation_agent(ALEX, ENRICHMENT, RISK, case_id="C-001", llm=llm)
    assert out["outcome"] == "approve"
    assert out["reason"] == "Low risk file."
    assert "review_checklist" not in out


def test_outcome_unchanged_when_llm_reason_fails() -> None:
    llm = FakeLLM({"recommendation": "not-json"})
    out = recommendation_agent(ALEX, ENRICHMENT, RISK, case_id="C-001", llm=llm)
    assert out["outcome"] == "approve"
    assert out["reason"] == "Score 0.38 mapped to approve by threshold."


def test_refer_includes_review_checklist() -> None:
    llm = FakeLLM(
        {
            "recommendation": {
                "outcome": "deny",
                "reason": "Refer for a human review of tobacco and labs.",
                "review_checklist": ["Confirm tobacco history", "Review abnormal labs"],
            }
        }
    )
    risk = {"score": 0.55, "band": "medium", "factors": ["Current tobacco use"]}
    out = recommendation_agent(ALEX, ENRICHMENT, risk, case_id="C-004", llm=llm)
    assert out["outcome"] == "refer"
    assert out["review_checklist"] == [
        "Confirm tobacco history",
        "Review abnormal labs",
    ]


def test_refer_fallback_keeps_outcome_and_checklist() -> None:
    llm = FakeLLM({"recommendation": "not-json"})
    risk = {"score": 0.55, "band": "medium", "factors": ["Current tobacco use"]}
    out = recommendation_agent(ALEX, ENRICHMENT, risk, case_id="C-004", llm=llm)
    assert out["outcome"] == "refer"
    assert out["reason"] == "Score 0.55 mapped to refer by threshold."
    assert out["review_checklist"]
