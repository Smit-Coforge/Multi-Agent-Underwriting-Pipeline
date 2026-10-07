from src.agents.recommendation import pick_outcome, recommendation_agent
from src.llm import FakeLLM
from tests.conftest import ALEX

ENRICHMENT = {
    "credit_band": "good",
    "prior_claims": 1,
    "high_value_flag": True,
    "summary": "Good credit.",
}
RISK = {"score": 0.38, "band": "medium_low", "factors": ["non-smoker"]}


def test_pick_outcome_thresholds() -> None:
    assert pick_outcome(0.38) == "approve"
    assert pick_outcome(0.4) == "refer"
    assert pick_outcome(0.7) == "refer"
    assert pick_outcome(0.81) == "deny"


def test_recommendation_agent_code_picks_outcome() -> None:
    llm = FakeLLM({"recommendation": {"outcome": "deny", "reason": "Low-to-medium risk."}})
    out = recommendation_agent(ALEX, ENRICHMENT, RISK, case_id="C-001", llm=llm)
    assert out["outcome"] == "approve"
    assert "reason" in out
