from src.agents.risk import compute_term_life_score, risk_scoring_agent
from src.llm import FakeLLM
from src.tools.enrichment_lookup import lookup_enrichment
from tests.conftest import ALEX, JORDAN

ENRICHMENT_ALEX = lookup_enrichment("C-001")
ENRICHMENT_JORDAN = lookup_enrichment("C-002")


def test_compute_score_low_risk_approvable() -> None:
    score, band, factors = compute_term_life_score(ALEX, ENRICHMENT_ALEX)
    assert score < 0.4
    assert band == "low"
    assert factors


def test_compute_score_high_risk_declinable() -> None:
    score, band, _factors = compute_term_life_score(JORDAN, ENRICHMENT_JORDAN)
    assert score > 0.7
    assert band == "high"


def test_llm_cannot_change_score_band_or_factors() -> None:
    llm = FakeLLM(
        {
            "risk": {
                "score": 0.99,
                "band": "high",
                "factors": ["tobacco"],
                "underwriter_note": "Model wanted a different score.",
            }
        }
    )
    out = risk_scoring_agent(ALEX, ENRICHMENT_ALEX, case_id="t", llm=llm)
    score, band, factors = compute_term_life_score(ALEX, ENRICHMENT_ALEX)
    assert out["score"] == score
    assert out["band"] == band
    assert out["factors"] == factors
    assert out["factors"] != ["tobacco"]
    assert out["underwriter_note"] == "Model wanted a different score."


def test_nondisclosure_unjustified_coverage_and_flags_raise_score() -> None:
    base, _, _ = compute_term_life_score(ALEX, ENRICHMENT_ALEX)

    disclosed = {
        **ENRICHMENT_ALEX,
        "nondisclosure_findings": [
            {
                "evidence_type": "prescription",
                "evidence": "metformin",
                "suggests": "type2_diabetes",
                "explanation": "not disclosed",
            }
        ],
    }
    nondisclosure_score, _, nondisclosure_factors = compute_term_life_score(ALEX, disclosed)
    assert nondisclosure_score > base
    assert any("non-disclosure" in factor.lower() for factor in nondisclosure_factors)

    unjustified = {
        **ENRICHMENT_ALEX,
        "coverage_assessment": {"justified": False, "reason": "20x income"},
    }
    unjustified_score, _, unjustified_factors = compute_term_life_score(ALEX, unjustified)
    assert unjustified_score > base
    assert any("not justified" in factor.lower() for factor in unjustified_factors)

    flagged = {
        **ALEX,
        "consistency_flags": [
            {
                "fields": ["occupation", "annual_income"],
                "issue": "Student income is implausible.",
            }
        ],
    }
    flagged_score, _, flagged_factors = compute_term_life_score(flagged, ENRICHMENT_ALEX)
    assert flagged_score > base
    assert any("consistency" in factor.lower() for factor in flagged_factors)

    many_flags = {
        **ALEX,
        "consistency_flags": [
            {"fields": ["age", "term_length_years"], "issue": f"flag {index}"}
            for index in range(6)
        ],
    }
    many_score, _, _ = compute_term_life_score(many_flags, ENRICHMENT_ALEX)
    assert many_score > flagged_score
    assert many_score - base <= 0.16 + 1e-9
