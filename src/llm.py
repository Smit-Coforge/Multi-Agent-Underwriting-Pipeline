from __future__ import annotations

import json
import os
from typing import Any

import ollama


class FakeLLM:
    """Canned JSON replies for tests. Same complete() shape as OllamaLLM."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5}

    def complete(self, messages: list[dict[str, str]], *, agent: str) -> str:
        payload = self.responses[agent]
        if isinstance(payload, str):
            return payload
        return json.dumps(payload)


class OllamaLLM:
    def __init__(
        self,
        host: str | None = None,
        model: str = "qwen3:8b",
    ) -> None:
        self._client = ollama.Client(
            host=host or os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
        )
        self.model = model
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def complete(self, messages: list[dict[str, str]], *, agent: str) -> str:
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            format="json",
            think=False,
            options={"num_predict": 220},
        )
        self.last_usage = {
            "prompt_tokens": int(resp.prompt_eval_count or 0),
            "completion_tokens": int(resp.eval_count or 0),
        }
        return resp.message.content or ""
