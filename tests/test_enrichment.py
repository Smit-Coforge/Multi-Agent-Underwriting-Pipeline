from src.agents.enrichment import enrichment_agent
from src.llm import FakeLLM
from tests.conftest import ALEX


def test_enrichment_agent_adds_lookups_and_summary() -> None:
    llm = FakeLLM({"enrichment": {"summary": "Good credit, one minor claim, high coverage."}})
    out = enrichment_agent(ALEX, case_id="C-001", llm=llm)
    assert out["credit_band"] == "good"
    assert out["prior_claims"] == 1
    assert out["high_value_flag"] is True
    assert "summary" in out
