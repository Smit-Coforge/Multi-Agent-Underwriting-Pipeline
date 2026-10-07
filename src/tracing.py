from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from src.state import CONTRACTS, CaseState

AgentFn = Callable[..., dict[str, Any]]


def write_trace(log: list[dict[str, Any]], record: dict[str, Any]) -> None:
    log.append(record)


def wrap(name: str, fn: AgentFn, llm: Any, log: list[dict[str, Any]]) -> Any:
    rule = CONTRACTS[name]

    def node(case: CaseState) -> dict[str, Any]:
        view = {key: case[key] for key in rule["reads"]}  # type: ignore[literal-required]
        case_id = case["case_id"]
        llm.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        for attempt in range(3):
            started_at = datetime.now(timezone.utc).isoformat()
            t0 = time.perf_counter()
            try:
                out = fn(view, case_id=case_id, llm=llm)
                if not isinstance(out, dict):
                    raise TypeError(f"{name} must return a dict")
                duration_ms = int((time.perf_counter() - t0) * 1000)
                usage = llm.last_usage
                write_trace(
                    log,
                    {
                        "case_id": case_id,
                        "agent": name,
                        "input": view,
                        "output": out,
                        "started_at": started_at,
                        "duration_ms": duration_ms,
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "cost_usd": 0,
                        "status": "ok",
                        "error": None,
                        "retry": attempt,
                    },
                )
                print(f"  ✓ {name:16} {duration_ms}ms")
                return {rule["writes"]: out}
            except Exception as exc:  # noqa: BLE001 — retry any agent failure
                duration_ms = int((time.perf_counter() - t0) * 1000)
                usage = llm.last_usage
                write_trace(
                    log,
                    {
                        "case_id": case_id,
                        "agent": name,
                        "input": view,
                        "output": None,
                        "started_at": started_at,
                        "duration_ms": duration_ms,
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "cost_usd": 0,
                        "status": "error",
                        "error": repr(exc),
                        "retry": attempt,
                    },
                )
                print(f"  ✗ {name:16} retry={attempt} {exc!r}")
                if attempt < 2:
                    time.sleep([1, 2][attempt])
        errors = list((case.get("meta") or {}).get("errors") or [])
        errors.append(name)
        retries = dict((case.get("meta") or {}).get("retries") or {})
        retries[name] = 3
        return {"meta": {"errors": errors, "retries": retries}}

    return node
