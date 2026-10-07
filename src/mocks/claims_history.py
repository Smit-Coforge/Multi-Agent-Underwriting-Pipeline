from typing import Any


def prior_claims_lookup(claims: list[Any] | None) -> int:
    return len(claims or [])
