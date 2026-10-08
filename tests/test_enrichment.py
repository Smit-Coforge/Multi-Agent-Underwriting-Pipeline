from src.agents.enrichment import enrichment_agent
from src.llm import FakeLLM
from src.tools.enrichment_lookup import FIELDS, lookup_enrichment
from tests.conftest import ALEX


def test_enrichment_agent_adds_lookups_and_summary() -> None:
    llm = FakeLLM({"enrichment": {"summary": "Clean labs and coverage consistent with income."}})
    looked_up = lookup_enrichment("C-001")
    out = enrichment_agent(ALEX, case_id="C-001", llm=llm)
    for key in FIELDS:
        assert out[key] == looked_up[key]
    assert out["summary"] == "Clean labs and coverage consistent with income."
