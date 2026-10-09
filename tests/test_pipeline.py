import json
from pathlib import Path

from src.agents.risk import compute_term_life_score
from src.graph import run_pipeline
from src.llm import FakeLLM
from src.tools.enrichment_lookup import lookup_enrichment
from tests.conftest import (
    ALEX,
    ALEX_RAW,
    CASEY_RAW,
    JORDAN,
    JORDAN_RAW,
    TimeoutOnRiskLLM,
    fake_for,
)


def _report(report_dir: Path, case_id: str) -> dict:
    return json.loads((report_dir / f"{case_id}.json").read_text(encoding="utf-8"))


def _load(folder: str, case_id: str) -> dict:
    return json.loads(Path(folder, f"{case_id}.json").read_text(encoding="utf-8"))


def test_low_risk_file_approves(tmp_path: Path) -> None:
    case = run_pipeline(ALEX_RAW, "C-001", fake_for(ALEX), report_dir=tmp_path)
    assert case["decision"]["outcome"] == "approve"
    assert case["risk"]["score"] < 0.4
    assert case["risk"]["factors"] != ["not the official factors"]
    report = _report(tmp_path, "C-001")
    assert report["case"]["case_id"] == "C-001"
    assert report["case"]["decision"]["outcome"] == "approve"
    agents = [row["agent"] for row in report["trace"]]
    assert agents == ["intake", "enrichment", "risk", "recommendation"]
    for row in report["trace"]:
        assert "input" in row
        assert "output" in row
        assert "duration_ms" in row
        assert "prompt_tokens" in row
        assert "completion_tokens" in row
        assert row["cost_usd"] == 0
        assert row["status"] == "ok"
        assert row["retry"] == 0


def test_medium_risk_file_refers(tmp_path: Path) -> None:
    raw = _load("data/applications", "C-004")
    llm = FakeLLM(
        {
            "intake": {
                "medical_conditions": ["high_cholesterol"],
                "consistency_flags": [],
            },
            "enrichment": {
                "nondisclosure_findings": [],
                "coverage_assessment": {
                    "justified": True,
                    "reason": "About 8 times income is within guidelines.",
                },
            },
            "risk": {"factors": ["tobacco"], "underwriter_note": "Smoker with elevated LDL."},
            "recommendation": {
                "outcome": "approve",
                "reason": "Refer because of tobacco use and elevated LDL.",
                "review_checklist": ["Confirm tobacco use", "Review LDL and statin"],
            },
        }
    )
    case = run_pipeline(raw, "C-004", llm, report_dir=tmp_path)
    assert case["decision"]["outcome"] == "refer"
    assert case["decision"]["review_checklist"]
    assert 0.4 <= case["risk"]["score"] <= 0.7
    assert case["risk"]["factors"] != ["tobacco"]


def test_high_risk_file_denies(tmp_path: Path) -> None:
    case = run_pipeline(JORDAN_RAW, "C-002", fake_for(JORDAN), report_dir=tmp_path)
    assert case["decision"]["outcome"] == "deny"
    assert case["risk"]["score"] > 0.7
    assert case["applicant"]["medical_conditions"] == ["copd", "hypertension"]
    assert case["applicant"]["occupation"] == "construction supervisor"


def test_risk_timeout_retries_then_escalates(tmp_path: Path) -> None:
    llm = TimeoutOnRiskLLM(
        {
            "intake": {"medical_conditions": [], "consistency_flags": []},
            "enrichment": {
                "nondisclosure_findings": [],
                "coverage_assessment": {
                    "justified": True,
                    "reason": "Within guidelines.",
                },
            },
            "recommendation": {"reason": "unused"},
        }
    )
    case = run_pipeline(CASEY_RAW, "C-003", llm, report_dir=tmp_path)
    assert case["decision"]["outcome"] == "refer"
    assert "risk failed after 3 attempts" in case["decision"]["reason"]
    assert case["risk"] is None
    rows = _report(tmp_path, "C-003")["trace"]
    risk_errors = [row for row in rows if row["agent"] == "risk"]
    assert [row["retry"] for row in risk_errors] == [0, 1, 2]
    assert all(row["status"] == "error" for row in risk_errors)
    assert all(row["cost_usd"] == 0 for row in rows)
    assert any(row["agent"] == "escalate" and row["status"] == "escalated" for row in rows)
    assert not any(row["agent"] == "recommendation" for row in rows)


def test_case_files_cover_scenarios() -> None:
    alex = _load("data/applications", "C-001")
    riley = _load("data/applications", "C-005")
    assert alex["name"] == "Alex Rivera"
    assert riley["name"] == "Riley Chen"
    for case_id, person in (("C-001", alex), ("C-005", riley)):
        score, _, _ = compute_term_life_score(
            {**person, "consistency_flags": []},
            lookup_enrichment(case_id),
        )
        assert score < 0.4

    jordan = _load("data/applications", "C-002")
    jordan_score, _, _ = compute_term_life_score(jordan, lookup_enrichment("C-002"))
    assert jordan["smoker"] is True
    assert jordan_score > 0.7

    nora = _load("data/applications", "C-007")
    assert nora["coverage_amount"] / nora["annual_income"] == 20

    helen = _load("data/applications", "C-008")
    assert helen["age"] + helen["term_length_years"] > 80
    assert helen["smoker"] is False
    assert any("copd" in str(item).lower() for item in helen["medical_conditions"])

    denise = _load("data/applications", "C-009")
    denise_file = _load("data/enrichment", "C-009")
    assert denise["medical_conditions"] == []
    assert denise_file["recent_lab_results"]["a1c"] == 7.0
    assert any(item["drug"] == "metformin" for item in denise_file["prescription_history"])

    evan = _load("data/applications", "C-010")
    assert evan["medical_conditions_text"] == "high BP, borderline sugar"

    for number in range(1, 11):
        record = _load("data/enrichment", f"C-{number:03d}")
        history = record["verified_medical_history"]
        financial = record["financial_coverage_check"]
        assert "matches_application" not in history
        assert "consistent" not in financial
        assert "coverage_to_income" in financial
        assert "existing_coverage_verified" in financial
