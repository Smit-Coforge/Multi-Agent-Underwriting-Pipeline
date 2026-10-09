import pytest

from src.agents.enrichment import enrichment_agent
from src.llm import FakeLLM
from src.tools.enrichment_lookup import FIELDS, lookup_enrichment
from tests.conftest import ALEX


def test_enrichment_rejects_evidence_not_in_records() -> None:
    llm = FakeLLM(
        {
            "enrichment": {
                "nondisclosure_findings": [
                    {
                        "evidence_type": "prescription",
                        "evidence": "metformin",
                        "suggests": "type2_diabetes",
                        "explanation": "This drug is not on the C-001 file.",
                    }
                ],
                "coverage_assessment": {
                    "justified": True,
                    "reason": "Face amount is within income guidelines.",
                },
            }
        }
    )
    with pytest.raises(ValueError):
        enrichment_agent(ALEX, case_id="C-001", llm=llm)


def test_enrichment_drops_a_normal_lab_cited_with_its_value() -> None:
    llm = FakeLLM(
        {
            "enrichment": {
                "nondisclosure_findings": [
                    {
                        "evidence_type": "lab",
                        "evidence": "a1c: 5.2",
                        "suggests": "prediabetes",
                        "explanation": "A1C was cited with its number.",
                    }
                ],
                "coverage_assessment": {
                    "justified": True,
                    "reason": "Face amount is within income guidelines.",
                },
            }
        }
    )
    out = enrichment_agent(ALEX, case_id="C-001", llm=llm)
    assert out["nondisclosure_findings"] == []


def test_enrichment_drops_finding_for_a_disclosed_condition() -> None:
    llm = FakeLLM(
        {
            "enrichment": {
                "nondisclosure_findings": [
                    {
                        "evidence_type": "prescription",
                        "evidence": "lisinopril",
                        "suggests": "hypertension",
                        "explanation": "Already on the application.",
                    },
                    {
                        "evidence_type": "lab",
                        "evidence": "a1c",
                        "suggests": "prediabetes",
                        "explanation": "A1C is in the prediabetes range and was not disclosed.",
                    },
                ],
                "coverage_assessment": {
                    "justified": True,
                    "reason": "Face amount fits income.",
                },
            }
        }
    )
    out = enrichment_agent(
        {"medical_conditions": ["hypertension"]},
        case_id="C-010",
        llm=llm,
    )
    assert out["nondisclosure_findings"] == [
        {
            "evidence_type": "lab",
            "evidence": "a1c",
            "suggests": "prediabetes",
            "explanation": "A1C is in the prediabetes range and was not disclosed.",
        }
    ]


def test_enrichment_stores_valid_finding() -> None:
    finding = {
        "evidence_type": "prescription",
        "evidence": "metformin",
        "suggests": "type2_diabetes",
        "explanation": "Metformin is on file and diabetes was not disclosed.",
    }
    llm = FakeLLM(
        {
            "enrichment": {
                "nondisclosure_findings": [finding],
                "coverage_assessment": {
                    "justified": True,
                    "reason": "Face amount fits income.",
                },
            }
        }
    )
    looked_up = lookup_enrichment("C-009")
    out = enrichment_agent({"medical_conditions": []}, case_id="C-009", llm=llm)
    assert out["nondisclosure_findings"] == [finding]
    assert out["coverage_assessment"]["justified"] is True
    for key in FIELDS:
        assert out[key] == looked_up[key]
    assert "summary" not in out
