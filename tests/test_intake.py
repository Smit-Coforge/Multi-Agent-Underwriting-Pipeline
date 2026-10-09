import pytest

from src.agents.intake import intake_agent
from src.llm import FakeLLM
from tests.conftest import ALEX_RAW


def test_intake_keeps_form_fields() -> None:
    llm = FakeLLM(
        {
            "intake": {
                "name": "Someone Else",
                "medical_conditions": [],
                "consistency_flags": [],
            }
        }
    )
    out = intake_agent(ALEX_RAW, case_id="C-001", llm=llm)
    assert out["name"] == "Alex Rivera"
    assert out["age"] == 34
    assert out["occupation"] == "software engineer"
    assert out["coverage_amount"] == 500000
    assert out["smoker"] is False
    assert out["medical_conditions"] == []
    assert out["annual_income"] == 85000
    assert out["consistency_flags"] == []


def test_intake_maps_free_text_to_allowed_labels() -> None:
    raw = {
        **ALEX_RAW,
        "medical_conditions": ["high blood pressure"],
        "medical_conditions_text": "high BP, borderline sugar",
    }
    llm = FakeLLM(
        {
            "intake": {
                "occupation": "astronaut",
                "medical_conditions": ["hypertension", "prediabetes"],
                "consistency_flags": [],
            }
        }
    )
    out = intake_agent(raw, case_id="C-010", llm=llm)
    assert out["medical_conditions"] == ["hypertension", "prediabetes"]
    assert out["medical_conditions_text"] == "high BP, borderline sugar"
    assert out["occupation"] == "software engineer"


def test_intake_rejects_unknown_condition_label() -> None:
    llm = FakeLLM(
        {"intake": {"medical_conditions": ["high BP"], "consistency_flags": []}}
    )
    with pytest.raises(ValueError):
        intake_agent(ALEX_RAW, case_id="C-001", llm=llm)


def test_intake_rejects_unknown_flag_field() -> None:
    llm = FakeLLM(
        {
            "intake": {
                "medical_conditions": [],
                "consistency_flags": [
                    {"fields": ["favorite_color"], "issue": "not on the form"}
                ],
            }
        }
    )
    with pytest.raises(ValueError):
        intake_agent(ALEX_RAW, case_id="C-001", llm=llm)


def test_intake_requires_form_fields() -> None:
    raw = {key: value for key, value in ALEX_RAW.items() if key != "smoker"}
    llm = FakeLLM({"intake": {"medical_conditions": [], "consistency_flags": []}})
    with pytest.raises(ValueError, match="smoker"):
        intake_agent(raw, case_id="C-001", llm=llm)
