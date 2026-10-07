from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.enrichment import enrichment_agent
from src.agents.intake import intake_agent
from src.agents.recommendation import recommendation_agent
from src.agents.risk import risk_scoring_agent
from src.state import CaseState, new_case
from src.tracing import wrap, write_trace


def _call_intake(view: dict[str, Any], *, case_id: str, llm: Any) -> dict[str, Any]:
    return intake_agent(view["raw"], case_id=case_id, llm=llm)


def _call_enrichment(view: dict[str, Any], *, case_id: str, llm: Any) -> dict[str, Any]:
    return enrichment_agent(view["applicant"], case_id=case_id, llm=llm)


def _call_risk(view: dict[str, Any], *, case_id: str, llm: Any) -> dict[str, Any]:
    return risk_scoring_agent(
        view["applicant"], view["enrichment"], case_id=case_id, llm=llm
    )


def _call_recommendation(view: dict[str, Any], *, case_id: str, llm: Any) -> dict[str, Any]:
    return recommendation_agent(
        view["applicant"],
        view["enrichment"],
        view["risk"],
        case_id=case_id,
        llm=llm,
    )


def _router(agent: str, nxt: str):
    def route(case: CaseState) -> str:
        errors = (case.get("meta") or {}).get("errors") or []
        if agent in errors:
            return "escalate"
        return nxt

    return route


def _escalate(case: CaseState, log: list[dict[str, Any]]) -> dict[str, Any]:
    errors = (case.get("meta") or {}).get("errors") or ["unknown"]
    agent = errors[0]
    decision = {
        "outcome": "refer",
        "reason": f"{agent} failed after 3 attempts",
    }
    write_trace(
        log,
        {
            "case_id": case["case_id"],
            "agent": "escalate",
            "input": {"meta": case.get("meta")},
            "output": decision,
            "started_at": "",
            "duration_ms": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0,
            "status": "escalated",
            "error": None,
            "retry": 0,
        },
    )
    print(f"  ⚠ escalate        REFER: {decision['reason']}")
    return {"decision": decision}


def build_graph(llm: Any, log: list[dict[str, Any]]) -> Any:
    graph = StateGraph(CaseState)
    graph.add_node("intake", wrap("intake", _call_intake, llm, log))
    graph.add_node("enrichment", wrap("enrichment", _call_enrichment, llm, log))
    graph.add_node("risk", wrap("risk", _call_risk, llm, log))
    graph.add_node(
        "recommendation",
        wrap("recommendation", _call_recommendation, llm, log),
    )
    graph.add_node("escalate", lambda case: _escalate(case, log))

    graph.add_edge(START, "intake")
    graph.add_conditional_edges(
        "intake",
        _router("intake", "enrichment"),
        {"enrichment": "enrichment", "escalate": "escalate"},
    )
    graph.add_conditional_edges(
        "enrichment",
        _router("enrichment", "risk"),
        {"risk": "risk", "escalate": "escalate"},
    )
    graph.add_conditional_edges(
        "risk",
        _router("risk", "recommendation"),
        {"recommendation": "recommendation", "escalate": "escalate"},
    )
    graph.add_edge("recommendation", END)
    graph.add_edge("escalate", END)
    return graph.compile()


def run_pipeline(
    raw: Any,
    case_id: str,
    llm: Any,
    report_dir: str | Path = "report",
) -> dict[str, Any]:
    log: list[dict[str, Any]] = []
    app = build_graph(llm, log)
    case = app.invoke(new_case(case_id, raw))
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{case_id}.json").write_text(
        json.dumps({"case": case, "trace": log}, indent=2, default=str),
        encoding="utf-8",
    )
    return case
