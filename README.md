# Multi-Agent Underwriting Pipeline

Auto insurance: intake → enrichment → risk → recommendation, with tracing and retry-then-escalate.

The pipeline rates the driver and the car: age, driving record, vehicle, and location. There are no labs and no diagnosis list. A bad record raises the price or leads to a decline.

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

New fields stay inside those keys: `violations` and `consistency_flags` on `applicant`, `nondisclosure_findings` and `rate_assessment` on `enrichment`, `underwriter_note` on `risk`, and `review_checklist` on a refer `decision`. An approve `decision` also has `rate_class` (`standard` or `surcharge`).

## Data

`data/applications/<case_id>.json` is the application: name, age, zip code, years licensed, vehicle year, make, and model, annual mileage, coverage level (`liability` or `full`), disclosed violations, and disclosed at-fault accident count. `driving_record_text` is optional free text from a "describe your record" box.

`data/enrichment/<case_id>.json` holds the outside record: motor vehicle record, claims, vehicle record, and garaging territory. It has no pre-baked match boolean. The enrichment agent always loads all four lookups.

| Case | What it is for |
|---|---|
| C-001 Alex Rivera | Clean record, standard rate |
| C-002 Jordan Blake | Decline: DUI, repeat at-fault accidents, high-risk zip |
| C-003 Casey Morgan | Clean record, nothing disclosed |
| C-004 Samir Patel | Young driver, high mileage, one speeding ticket |
| C-005 Riley Chen | Clean record, standard rate |
| C-006 Harold Nguyen | Suspended license and reckless driving |
| C-007 Nora Alvarez | 26,000 miles a year, so the price goes up |
| C-008 Helen Park | Licensed longer than her age allows; 2008 car with full coverage |
| C-009 Denise Okonkwo | Nothing disclosed; speeding on the MVR |
| C-010 Evan Brooks | Free text: "speeding ticket and a fender bender" |

## Agents

The LLM proposes the labels, flags, findings, surcharge, score, band, factors, note, outcome, and reason. Code raises `ValueError` when that proposal fails one of the checks below. The node wrapper then retries that agent 3 times and refers. A failed reply is kept on the trace as `reply`.

| Agent | LLM proposes | Code raises unless |
|---|---|---|
| intake | Allowed violation labels from `violations_disclosed` and `driving_record_text`, and `consistency_flags`. The model's issue text is what gets stored. | Form fields are copied from the JSON. The label set matches the form (`speeding`, `red_light`, `dui`, `reckless`, `at_fault_accident`, `suspended_license`, `other`). A flag is kept only when its fields are real application fields and that check is true on the form. The checks are: `years_licensed > age - 16`; `age < 25` and `annual_mileage >= 20000`; `vehicle_year < 2010` and `coverage_level` is `full`; `accidents_disclosed > years_licensed`. A flag that fails its check or names an unknown field raises. A check the form meets that the model omitted raises. Required keys still have to be present. |
| enrichment | `nondisclosure_findings` (`evidence_type` of `mvr`, plus evidence, suggests, and an explanation) and `rate_assessment` (`surcharge` plus a reason). | All four lookups are copied from the file. Evidence is a string on `mvr.violations`, or `at_fault_accidents` when that count is higher than `accidents_disclosed`. `suggests` is that same label. A finding for a violation already disclosed is dropped. The kept labels must be exactly the undisclosed ones. `surcharge` must match the record: high-risk zip, license not valid, 2 or more at-fault accidents, mileage 25000 or more, or a `dui`, `reckless`, or `suspended_license` on the application or the MVR. The reason must be non-empty. |
| risk | `score`, `band`, `factors`, and `underwriter_note`. | `score` is between 0 and 1. `band` matches that score. `factors` is a non-empty list of non-empty strings. `underwriter_note` is a non-empty string. A file with no DUI has no factor that uses the word dui. |
| recommendation | `outcome`, `reason`, `rate_class` on approve, and, on refer, `review_checklist`. | `outcome` is approve, refer, or deny. `reason` is non-empty. An approve uses `surcharge` when the rate assessment says so, and `standard` otherwise. A refer includes a non-empty `review_checklist`. |

## How to run

```bash
# unit + pipeline tests (FakeLLM, no Ollama)
pytest tests

# live case (Ollama qwen3:8b on the host)
python run.py C-001
# C-002 through C-010
```

Risk stores the model's score when it is between 0 and 1, the band matches that score, the factors and note are non-empty, and a file with no DUI has no dui factor. Recommendation stores the model's outcome when it is approve, refer, or deny, the reason is non-empty, an approve carries the rate class the assessment requires, and a refer includes a checklist.

If an agent raises (timeout, bad JSON, or a proposal that disagrees with the file), the wrapper retries 3 times then refers. Nothing is keyed off a special case id.

Docker talks to host Ollama at `http://host.docker.internal:11434` (`OLLAMA_HOST` in `docker-compose.yml`). Each run writes `report/<case_id>/<timestamp>.json` (the case dict plus the step trace).

## Part 4

`cost_usd` stays 0 because `token_cost_usd` prices a local Ollama model at 0. A paid API would be the prompt and completion counts times that provider's rate.

A failed check does not change the decision by itself. The wrapper retries that agent 3 times, keeps each raw reply on the trace, and then refers the case to a person.
