"""LLM plumbing. No network, no money: `send` is always a fake."""

import pytest

from pytest_triage.llm import (
    BadOutput,
    OverBudget,
    call,
    cost_usd,
    generate_json,
    json_body,
    output_text,
)

# Shape of a real recorded Interactions response, trimmed.
RESPONSE = {
    "id": "v1_abc",
    "status": "completed",
    "model": "gemini-3.5-flash-lite",
    "usage": {
        "total_input_tokens": 1000,
        "total_cached_tokens": 400,
        "total_output_tokens": 100,
        "total_thought_tokens": 50,
    },
    "steps": [
        {"type": "thought", "signature": "El4K..."},
        {
            "type": "model_output",
            "content": [
                {"type": "text", "text": '{"labels": '},
                {"type": "text", "text": '["type: bug"]}'},
            ],
        },
    ],
}


def test_cost_bills_cached_at_cached_rate_and_thoughts_as_output():
    expected = (600 * 0.30 + 400 * 0.03 + 150 * 2.50) / 1_000_000
    assert cost_usd("gemini-3.5-flash-lite", RESPONSE["usage"]) == pytest.approx(expected)


def test_unknown_model_refuses_to_guess():
    with pytest.raises(KeyError):
        cost_usd("gemini-9-ultra", RESPONSE["usage"])


def test_output_text_joins_text_and_skips_thoughts():
    assert output_text(RESPONSE) == '{"labels": ["type: bug"]}'
    assert output_text({"steps": [{"type": "thought"}]}) == ""


def fake_send(calls, response=RESPONSE):
    def send(body):
        calls.append(body)
        return response

    return send


def test_identical_calls_are_paid_once(tmp_path):
    calls = []
    kwargs = dict(send=fake_send(calls), cache_dir=tmp_path / "c", ledger=tmp_path / "l.jsonl")
    call({"model": "gemini-3.5-flash-lite", "input": "hi"}, **kwargs)
    call({"input": "hi", "model": "gemini-3.5-flash-lite"}, **kwargs)  # key order irrelevant
    assert len(calls) == 1
    assert len((tmp_path / "l.jsonl").read_text().splitlines()) == 1


def test_cap_refuses_before_sending(tmp_path):
    calls = []
    with pytest.raises(OverBudget):
        call(
            {"input": "hi"},
            send=fake_send(calls),
            cache_dir=tmp_path / "c",
            ledger=tmp_path / "l.jsonl",
            cap=0.0,
        )
    assert calls == []


def test_json_body_is_stateless_and_capped():
    body = json_body("sys", "hello", {"type": "object"}, max_output_tokens=64)
    assert body["store"] is False
    assert body["generation_config"]["max_output_tokens"] == 64
    assert body["response_format"]["mime_type"] == "application/json"


def test_incomplete_responses_are_not_cached_and_raise(tmp_path):
    calls = []
    incomplete = RESPONSE | {"status": "incomplete"}
    kwargs = dict(
        send=fake_send(calls, incomplete), cache_dir=tmp_path / "c", ledger=tmp_path / "l.jsonl"
    )
    call({"input": "hi"}, **kwargs)
    call({"input": "hi"}, **kwargs)
    assert len(calls) == 2  # retried, not replayed from cache
    with pytest.raises(BadOutput):
        generate_json({}, send=lambda b: incomplete)


def test_thinking_models_get_their_own_level_and_headroom():
    lite = json_body("s", "i", {}, model="gemini-3.5-flash-lite", max_output_tokens=64)
    flash = json_body("s", "i", {}, model="gemini-3.8-flash", max_output_tokens=64)
    assert lite["generation_config"] == {"max_output_tokens": 64, "thinking_level": "minimal"}
    assert flash["generation_config"]["thinking_level"] == "low"
    assert flash["generation_config"]["max_output_tokens"] > 64


def test_generate_json_parses_or_raises():
    data, _ = generate_json({}, send=lambda b: RESPONSE)
    assert data == {"labels": ["type: bug"]}
    broken = RESPONSE | {
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": "nope"}]}]
    }
    with pytest.raises(BadOutput):
        generate_json({}, send=lambda b: broken)
