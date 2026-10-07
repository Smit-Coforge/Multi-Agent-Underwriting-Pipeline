# Multi-Agent Underwriting Pipeline

Week 8 lab: intake → enrichment → risk → recommendation, with tracing and retry-then-escalate.

## Part 1: State and memory

**Choice: shared case file (one dict per run).** Not message passing.

Each run builds a fresh dict inside `run_pipeline()` and every agent reads from / writes to that dict. We are not threading each agent’s output in as the next agent’s only input.

**Why**

1. Observability — at any moment the whole case is in one place, which is what the trace logs.
2. Extensibility — a later agent (for example a fraud check) can read `applicant` and `risk` without re-wiring payloads between existing steps.

**Guardrails (shared state is riskier if left open)**

| Guardrail | How |
|---|---|
| Fresh case per run | `new_case()` inside `run_pipeline()`. Never a module-level `CASE = {}`. |
| `case_id` everywhere | On every agent call, trace line, and log line. |
| Read/write contracts | The node wrapper in `src/tracing.py` passes only allowed read keys and returns only the agent’s write key. |
| Leakage test | `tests/test_no_leakage.py` runs case A then case B and asserts B contains none of A. |

The file under `report/` is observability, not working memory. Agents never read it.

### Case keys

| Key | Written by | Read by |
|---|---|---|
| `case_id` | `run_pipeline` | everyone (logs / traces) |
| `raw` | `run_pipeline` | intake |
| `applicant` | intake | enrichment, risk, recommendation |
| `enrichment` | enrichment | risk, recommendation |
| `risk` | risk | recommendation |
| `decision` | recommendation, or escalate on failure | final output |
| `meta` | node wrapper | failure routing, tests |

### Read/write matrix

`R` = may read. `W` = may write. Anything else is blocked by the wrapper.

| Agent | raw | applicant | enrichment | risk | decision |
|---|---|---|---|---|---|
| intake | R | W | | | |
| enrichment | | R | W | | |
| risk | | R | R | W | |
| recommendation | | R | R | R | W |

## How to run

```bash
# unit + pipeline tests (FakeLLM, no Ollama)
pytest tests

# live case (Ollama qwen3:8b on the host)
python run.py C-001
python run.py C-002
python run.py C-003
```

Term life score is computed in code (tobacco, age, credit band, claims, face/income). The LLM only lists factors and writes the decision reason. Thresholds: `< 0.4` approve, `≤ 0.7` refer, else deny.

If an agent raises (timeout, bad JSON), the wrapper retries 3 times then refers. Nothing is keyed off a special case id.

Docker talks to host Ollama at `http://host.docker.internal:11434` (`OLLAMA_HOST` in `docker-compose.yml`). Each run writes `report/<case_id>.json` (full case dict plus step trace). That folder is gitignored.

