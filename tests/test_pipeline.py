import json
from pathlib import Path

from src.graph import run_pipeline
from tests.conftest import (
    ALEX,
    ALEX_RAW,
    CASEY,
    CASEY_RAW,
    JORDAN,
    JORDAN_RAW,
    TimeoutOnRiskLLM,
    fake_for,
)


def _report(report_dir: Path, case_id: str) -> dict:
    return json.loads((report_dir / f"{case_id}.json").read_text(encoding="utf-8"))


def test_low_risk_file_approves(tmp_path) -> None:
    case = run_pipeline(ALEX_RAW, "C-001", fake_for(ALEX), report_dir=tmp_path)
    assert case["decision"]["outcome"] == "approve"
    report = _report(tmp_path, "C-001")
    assert report["case"]["case_id"] == "C-001"
    assert report["case"]["decision"]["outcome"] == "approve"
    agents = [row["agent"] for row in report["trace"]]
    assert agents == ["intake", "enrichment", "risk", "recommendation"]
    for row in report["trace"]:
        assert "duration_ms" in row
        assert "prompt_tokens" in row
        assert "completion_tokens" in row
        assert "input" in row
        assert "output" in row


def test_high_risk_file_denies(tmp_path) -> None:
    case = run_pipeline(JORDAN_RAW, "C-002", fake_for(JORDAN), report_dir=tmp_path)
    assert case["decision"]["outcome"] == "deny"


def test_risk_timeout_retries_then_escalates(tmp_path) -> None:
    llm = TimeoutOnRiskLLM(
        {
            "intake": CASEY,
            "enrichment": {"summary": "Term life file summary."},
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
    assert any(row["agent"] == "escalate" and row["status"] == "escalated" for row in rows)
    assert not any(row["agent"] == "recommendation" for row in rows)
