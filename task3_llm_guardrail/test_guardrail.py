"""Run:  pytest -q task3_llm_guardrail"""

import asyncio
import json
import random
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import guardrail_gateway as gateway  # noqa: E402
import mock_llm  # noqa: E402
from redactor import MAX_HOLDBACK, REDACTED, StreamRedactor, redact  # noqa: E402

PII = ["john.doe+billing@example.co.uk", "123-45-6789", "4111 1111 1111 1111", "5500-0000-0000-0004"]


def run_stream(text, sizes):
    r, out, i = StreamRedactor(), [], 0
    rnd = random.Random(sizes)
    while i < len(text):
        n = rnd.randint(1, 8)
        out.append(r.feed(text[i : i + n]))
        i += n
        assert r.pending <= MAX_HOLDBACK + 8
    out.append(r.flush())
    return "".join(out)


# ---- redactor unit tests -------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("mail a.b@c.io now", f"mail {REDACTED} now"),
        ("ssn 123-45-6789.", f"ssn {REDACTED}."),
        ("ssn 123 45 6789", f"ssn {REDACTED}"),
        ("ssn 123456789", f"ssn {REDACTED}"),
        ("invalid ssn 000-12-3456", "invalid ssn 000-12-3456"),
        ("card 4111111111111111", f"card {REDACTED}"),
        ("card 4111-1111-1111-1111!", f"card {REDACTED}!"),
        ("order 1234567890123", "order 1234567890123"),  # fails Luhn, left alone
        ("phone 2024", "phone 2024"),
    ],
)
def test_redact(text, expected):
    assert redact(text) == expected


@pytest.mark.parametrize("seed", range(200))
def test_stream_matches_batch_for_any_chunking(seed):
    text = mock_llm.REPLY
    got = run_stream(text, seed)
    assert got == redact(text)
    for p in PII:
        assert p not in got
    assert "1234567890123" in got


def test_char_by_char():
    r = StreamRedactor()
    out = "".join(r.feed(c) for c in mock_llm.REPLY) + r.flush()
    assert out == redact(mock_llm.REPLY)


def test_emits_early_not_at_end():
    r = StreamRedactor()
    assert r.feed("Hello there ") == "Hello there "
    assert r.feed("john") == ""          # might become an email
    assert r.feed("@x.io and") == f"{REDACTED} "
    assert r.flush() == "and"


def test_memory_bounded_on_giant_token():
    r = StreamRedactor()
    emitted = "".join(r.feed("a" * 50) for _ in range(100))
    assert r.pending <= MAX_HOLDBACK + 50
    assert len(emitted) > 4000


# ---- end to end through the gateway -------------------------------------

@pytest.fixture
async def client(monkeypatch):
    monkeypatch.setattr(gateway, "UPSTREAM_URL", "http://mock/v1/chat/completions")
    monkeypatch.setattr(mock_llm, "CHUNK_DELAY", 0)
    gateway.app.state.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_llm.app))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url="http://gw") as c:
        yield c
    await gateway.app.state.http.aclose()


def parse_sse(text):
    events = [l[5:].strip() for l in text.splitlines() if l.startswith("data:")]
    assert events[-1] == "[DONE]"
    return [json.loads(e) for e in events[:-1]]


@pytest.mark.parametrize("seed", [1, 2, 3, 42])
async def test_e2e_stream(client, seed):
    r = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": True, "seed": seed})
    assert r.headers["content-type"].startswith("text/event-stream")
    chunks = parse_sse(r.text)
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == redact(mock_llm.REPLY)
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(c["id"] == "chatcmpl-mock" for c in chunks)


async def test_e2e_non_stream(client):
    r = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.json()["choices"][0]["message"]["content"] == redact(mock_llm.REPLY)


async def test_bad_request(client):
    r = await client.post("/v1/chat/completions", json={"nope": 1})
    assert r.status_code == 400


# ---- TTFT: first token goes out while upstream is still generating ---------

class FakeUpstream:
    def __init__(self, parts, delay):
        self.parts, self.delay = parts, delay

    async def aiter_lines(self):
        for p in self.parts:
            await asyncio.sleep(self.delay)
            yield "data: " + json.dumps({"id": "x", "choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]})
        yield "data: [DONE]"

    async def aclose(self):
        pass


async def test_ttft_is_not_blocked_by_full_response():
    parts = ["Hi ", "there, "] + ["word "] * 50
    start = time.perf_counter()
    gen = gateway.redact_stream(FakeUpstream(parts, delay=0.02))
    first = await gen.__anext__()
    ttft = time.perf_counter() - start
    assert b"Hi " in first
    assert ttft < 0.1  # full stream takes > 1s
    async for _ in gen:
        pass
