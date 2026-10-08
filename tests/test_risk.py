from src.agents.risk import compute_term_life_score, risk_scoring_agent
from src.llm import FakeLLM
from src.tools.enrichment_lookup import lookup_enrichment
from tests.conftest import ALEX, JORDAN

ENRICHMENT_ALEX = {
    **lookup_enrichment("C-001"),
    "summary": "Clean labs and consistent coverage.",
}
ENRICHMENT_JORDAN = {
    **lookup_enrichment("C-002"),
    "summary": "Smoker with COPD, abnormal labs, coverage above income.",
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
