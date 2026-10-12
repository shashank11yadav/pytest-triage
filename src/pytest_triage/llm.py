"""The one door to Gemini. Every LLM call goes through `call()`:

  CACHE   identical request bodies are never paid for twice (data/llm_cache/)
  LEDGER  every paid call is recorded with its cost (data/ledger.jsonl)
  CAP     once total spend reaches SPEND_CAP_USD, calls are refused before sending

Requests are stateless (`store: false`): each issue is triaged independently,
so server-side history buys nothing and would retain strangers' text at Google.
Explicit caching isn't available on the Interactions API; implicit caching is
automatic and shows up as usage.total_cached_tokens.

Docs (checked 2026-10-12):
    overview   https://ai.google.dev/gemini-api/docs/interactions
    reference  https://ai.google.dev/api/interactions-api
    pricing    https://ai.google.dev/gemini-api/docs/pricing
    caching    https://ai.google.dev/gemini-api/docs/caching
"""

import hashlib
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx

from pytest_triage.config import (
    GEMINI_URL,
    LEDGER,
    LLM_CACHE,
    PRICES,
    SPEND_CAP_USD,
    THINKING,
    THINKING_HEADROOM,
    WORKHORSE,
    gemini_key,
)

RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4


class OverBudget(Exception):
    """Raised instead of making a call the spend cap forbids."""


class BadOutput(Exception):
    """The model returned something that isn't the JSON we asked for."""


def cache_key(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def cost_usd(model: str, usage: dict) -> float:
    """Dollars for one call. Cached input is cheaper but not free; thinking bills as output."""
    price = PRICES[model]  # KeyError on an unknown model: never guess a price
    cached = usage.get("total_cached_tokens", 0)
    fresh = usage.get("total_input_tokens", 0) - cached
    out = usage.get("total_output_tokens", 0) + usage.get("total_thought_tokens", 0)
    return (fresh * price["input"] + cached * price["cached"] + out * price["output"]) / 1_000_000


def output_text(response: dict) -> str:
    """Text of every text item in every model_output step (thought steps carry none)."""
    return "".join(
        item.get("text", "")
        for step in response.get("steps", [])
        if step.get("type") == "model_output"
        for item in step.get("content", [])
        if item.get("type") == "text"
    )


def spent(ledger: Path = LEDGER) -> float:
    if not ledger.exists():
        return 0.0
    with ledger.open() as f:
        return sum(json.loads(line)["cost_usd"] for line in f)


def post(body: dict, sleep: Callable[[float], None] = time.sleep) -> dict:
    """The only function that touches the network. Retries transient errors with backoff."""
    for attempt in range(MAX_ATTEMPTS):
        response = httpx.post(
            GEMINI_URL, headers={"x-goog-api-key": gemini_key()}, json=body, timeout=120.0
        )
        if response.status_code not in RETRY_STATUSES or attempt == MAX_ATTEMPTS - 1:
            if response.is_error:  # surface the API's own message, e.g. an unsupported setting
                raise RuntimeError(f"Gemini {response.status_code}: {response.text[:500]}")
            return response.json()
        sleep(2**attempt * 2)  # 2s, 4s, 8s
    raise AssertionError("unreachable")


def call(
    body: dict,
    send: Callable[[dict], dict] = post,
    cache_dir: Path = LLM_CACHE,
    ledger: Path = LEDGER,
    cap: float = SPEND_CAP_USD,
) -> dict:
    """Cached, recorded, capped Gemini call. Returns the raw response."""
    path = cache_dir / f"{cache_key(body)}.json"
    if path.exists():
        return json.loads(path.read_text())

    if (total := spent(ledger)) >= cap:
        raise OverBudget(f"spent ${total:.4f} of the ${cap:.2f} cap")

    response = send(body)

    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as f:
        row = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": response["model"],
            "usage": response["usage"],
            "cost_usd": cost_usd(response["model"], response["usage"]),
            "store": body.get("store", False),
        }
        f.write(json.dumps(row) + "\n")

    # Only finished answers are cached: an "incomplete" one (output cut off by
    # max_output_tokens) would otherwise be replayed forever.
    if response.get("status") == "completed":
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(response))
    return response


def json_body(
    system: str,
    user_input: str,
    schema: dict,
    model: str = WORKHORSE,
    max_output_tokens: int = 256,
    thinking_level: str | None = None,
) -> dict:
    """A stateless structured-output request. Output is capped: on the workhorse,
    output tokens cost 8.3x input tokens, so asking for less back is the first lever."""
    return {
        "model": model,
        "system_instruction": system,
        "input": user_input,
        "store": False,
        "response_format": {"type": "text", "mime_type": "application/json", "schema": schema},
        "generation_config": {
            "max_output_tokens": max_output_tokens + THINKING_HEADROOM.get(model, 0),
            "thinking_level": thinking_level or THINKING.get(model, "low"),
        },
    }


def generate_json(body: dict, send: Callable[[dict], dict] = call) -> tuple[dict, dict]:
    """(parsed JSON, raw response). Raises BadOutput if the text isn't valid JSON."""
    response = send(body)
    if response.get("status") != "completed":
        raise BadOutput(f"status {response.get('status')!r} (output cut off or failed)")
    text = output_text(response)
    try:
        return json.loads(text), response
    except json.JSONDecodeError as e:
        raise BadOutput(f"not JSON: {text[:200]!r}") from e
