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

New fields stay inside those keys: `medical_conditions` and `consistency_flags` on `applicant`, `nondisclosure_findings` and `coverage_assessment` on `enrichment`, `underwriter_note` on `risk`, and `review_checklist` on a refer `decision`.

## Data

`data/applications/<case_id>.json` is the application form: name, age, gender, occupation (typed text), annual income, coverage amount, term length, smoker / nicotine use, medical conditions, and existing life insurance. `medical_conditions_text` is optional free text from a "describe your conditions" box.

`data/enrichment/<case_id>.json` holds the raw record: prescription history, recent lab results, conditions on file, `coverage_to_income`, and `existing_coverage_verified`. It has no pre-baked match or consistency boolean. The enrichment agent always loads all four lookups.

| Case | What it is for |
|---|---|
| C-001 Alex Rivera | Clean approve |
| C-002 Jordan Blake | Deny: smoker, COPD, abnormal labs |
| C-003 Casey Morgan | Teacher, nothing disclosed |
| C-004 Samir Patel | Smoker with high cholesterol |
| C-005 Riley Chen | Clean approve |
| C-006 Harold Nguyen | Diabetes and coronary artery disease |
| C-007 Nora Alvarez | Coverage 20× income |
| C-008 Helen Park | Age 62 asking for a 30-year term; non-smoker with COPD |
| C-009 Denise Okonkwo | Nothing disclosed; metformin and A1C 7.0 on file |
| C-010 Evan Brooks | Free text: "high BP, borderline sugar" |

## Agents

The LLM proposes. Code checks the proposal against the form, the looked-up records, or an allowed list before writing it onto the case. A failed check raises `ValueError`. The node wrapper retries that agent 3 times, then refers. Code computes the score, the factor list, and the approve / refer / deny outcome.

| Agent | LLM proposes | Code checks and keeps |
|---|---|---|
| intake | Allowed condition labels from `medical_conditions` and `medical_conditions_text`. `consistency_flags` whose `fields` are real form fields. | Form fields copied from the JSON. Every label must be in the allowed list (`hypertension`, `prediabetes`, `type2_diabetes`, `high_cholesterol`, `asthma`, `depression_anxiety`, `sleep_apnea`, `cad`, `copd`, `cancer_history`, `stroke_history`, `other`). A flag is kept only when that check is true on the form. Required keys still have to be present. |
| enrichment | `nondisclosure_findings` (`evidence_type` of `prescription`, `lab`, or `medical_history`) and `coverage_assessment` (`justified` plus a reason). | All four lookups. Drug names and history conditions must match exactly. A lab value such as `a1c: 5.2` is read as the key `a1c`, then dropped unless that value supports the label. A finding whose label is already disclosed is dropped. `justified` must be a bool. |
| risk | An `underwriter_note` only. | `compute_term_life_score` sets `score`, `band`, and `factors`. A non-empty findings list adds 0.12. `justified: false` adds 0.18. Each consistency flag adds 0.08, capped at 0.16. |
| recommendation | A reason that uses the flags, findings, coverage assessment, and factors. On refer, a short `review_checklist`. | `pick_outcome(score)`. If the reason call fails, the outcome stays and the fallback sentence is used. Refer still includes a checklist. |

## How to run

```bash
# unit + pipeline tests (FakeLLM, no Ollama)
pytest tests

# live case (Ollama qwen3:8b on the host)
python run.py C-001
# C-002 through C-010
```

Term life debits are computed in code: tobacco, age, disclosed conditions, severe conditions, lab flags, prescriptions, face amount versus income, non-disclosure findings, unjustified coverage, and consistency flags. Thresholds: `< 0.4` approve, `≤ 0.7` refer, else deny.

If an agent raises (timeout, bad JSON), the wrapper retries 3 times then refers. Nothing is keyed off a special case id.

Docker talks to host Ollama at `http://host.docker.internal:11434` (`OLLAMA_HOST` in `docker-compose.yml`). Each run writes `report/<case_id>.json` (full case dict plus step trace). That folder is gitignored.

