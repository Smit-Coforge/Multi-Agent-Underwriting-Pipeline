from src.agents.intake import intake_agent
from src.llm import FakeLLM
from tests.conftest import ALEX, ALEX_RAW


def test_intake_agent_extracts_applicant() -> None:
    llm = FakeLLM({"intake": ALEX})
    out = intake_agent(ALEX_RAW, case_id="C-001", llm=llm)
    assert out["name"] == "Alex Rivera"
    assert out["age"] == 34
    assert out["coverage_amount"] == 500000
    assert out["smoker"] is False
    assert out["medical_conditions"] == []
    assert out["annual_income"] == 85000
