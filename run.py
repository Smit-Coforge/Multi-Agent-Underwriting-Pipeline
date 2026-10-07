from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.graph import run_pipeline
from src.llm import OllamaLLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the underwriting pipeline")
    parser.add_argument("case_id")
    args = parser.parse_args()
    path = Path("data/applications") / f"{args.case_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    print(f"▶ {args.case_id}")
    case = run_pipeline(raw, args.case_id, llm=OllamaLLM())
    decision = case.get("decision") or {}
    outcome = str(decision.get("outcome", "?")).upper()
    print(f"  decision  {outcome}: {decision.get('reason', '')}")
    print(f"  report → report/{args.case_id}.json")


if __name__ == "__main__":
    main()
