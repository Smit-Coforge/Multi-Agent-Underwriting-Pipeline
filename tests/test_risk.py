from src.agents.risk import compute_term_life_score, risk_scoring_agent
from src.llm import FakeLLM
from tests.conftest import ALEX, JORDAN

ENRICHMENT_ALEX = {
    "credit_band": "good",
    "prior_claims": 1,
    "high_value_flag": True,
    "summary": "Good credit, one minor claim.",
}
ENRICHMENT_JORDAN = {
    "credit_band": "poor",
    "prior_claims": 3,
    "high_value_flag": True,
    "summary": "Poor credit, three claims.",
}


def test_compute_score_low_risk_approvable() -> None:
    score, band, factors = compute_term_life_score(ALEX, ENRICHMENT_ALEX)
    assert score < 0.4
    assert band == "low"
    assert factors


def test_compute_score_high_risk_declinable() -> None:
    score, band, _factors = compute_term_life_score(JORDAN, ENRICHMENT_JORDAN)
    assert score > 0.7
    assert band == "high"


def test_risk_scoring_agent_uses_computed_score_not_llm_score() -> None:
    llm = FakeLLM({"risk": {"score": 0.99, "factors": ["tobacco"]}})
    out = risk_scoring_agent(ALEX, ENRICHMENT_ALEX, case_id="t", llm=llm)
    assert out["score"] < 0.4
    assert out["factors"] == ["tobacco"]
