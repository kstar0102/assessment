"""
Task 3: LLM gateway with a real time PII redaction guardrail.

POST /v1/chat/completions   (OpenAI compatible request/response shape)

  client <--SSE-- gateway <--SSE-- provider
                    |
              StreamRedactor per choice index

  * Upstream SSE is read line by line with httpx aiter_lines, nothing is
    buffered beyond the redactor's small holdback tail.
  * Each upstream chunk is re emitted with the same envelope (id, model,
    created...) and only delta.content swapped for the redacted text.
  * Chunks that carry no content (role, finish_reason, tool calls) pass
    straight through so TTFT isn't held hostage by the guardrail.
  * Before a finish_reason or [DONE] we flush the holdback so no text is lost.
  * If the client disconnects the generator is closed, which closes the
    upstream stream too (no orphaned provider requests).
  * Non streaming requests are supported and redacted in one pass.

Env
  UPSTREAM_URL      default http://127.0.0.1:9100/v1/chat/completions (mock)
  UPSTREAM_API_KEY  sent as Bearer to the provider

Run:  uvicorn guardrail_gateway:app --port 8081
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from .redactor import StreamRedactor, redact
except ImportError:
    from redactor import StreamRedactor, redact

log = logging.getLogger("llm-guardrail")

UPSTREAM_URL = os.environ.get("UPSTREAM_URL", "http://127.0.0.1:9100/v1/chat/completions")
UPSTREAM_API_KEY = os.environ.get("UPSTREAM_API_KEY", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # read timeout is per chunk, so long generations are fine as long as tokens keep coming
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0))
    yield
    await app.state.http.aclose()


app = FastAPI(title="LLM Gateway PII Guardrail", lifespan=lifespan)


def sse(obj: dict | str) -> bytes:
    data = obj if isinstance(obj, str) else json.dumps(obj, separators=(",", ":"))
    return f"data: {data}\n\n".encode()


def gateway_error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"type": "gateway_error", "code": code, "message": message}}, status_code=status)


def _upstream_headers() -> dict:
    h = {"content-type": "application/json", "accept": "text/event-stream"}
    if UPSTREAM_API_KEY:
        h["authorization"] = f"Bearer {UPSTREAM_API_KEY}"
    return h


async def redact_stream(upstream: httpx.Response) -> AsyncIterator[bytes]:
    redactors: dict[int, StreamRedactor] = {}
    template: dict | None = None  # last chunk envelope, used for final flushes

    def flush_chunk(index: int) -> bytes | None:
        r = redactors.get(index)
        tail = r.flush() if r else ""
        if not tail or template is None:
            return None
        chunk = {**template, "choices": [{"index": index, "delta": {"content": tail}, "finish_reason": None}]}
        return sse(chunk)

    try:
        async for line in upstream.aiter_lines():
            if not line.startswith("data:"):
                continue  # comments, keep alives, event: lines
            data = line[5:].strip()
            if data == "[DONE]":
                for idx in list(redactors):
                    out = flush_chunk(idx)
                    if out:
                        yield out
                yield sse("[DONE]")
                return
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                log.warning("dropping malformed upstream chunk")
                continue

            template = {k: v for k, v in chunk.items() if k != "choices"}
            for choice in chunk.get("choices") or []:
                idx = choice.get("index", 0)
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    safe = redactors.setdefault(idx, StreamRedactor()).feed(content)
                    if safe:
                        yield sse({**template, "choices": [{**choice, "delta": {**delta, "content": safe}, "finish_reason": None}]})
                    elif set(delta) - {"content"}:  # keep role etc even if text held back
                        yield sse({**template, "choices": [{**choice, "delta": {k: v for k, v in delta.items() if k != "content"}, "finish_reason": None}]})
                    if choice.get("finish_reason"):
                        out = flush_chunk(idx)
                        if out:
                            yield out
                        yield sse({**template, "choices": [{"index": idx, "delta": {}, "finish_reason": choice["finish_reason"]}]})
                else:
                    if choice.get("finish_reason"):
                        out = flush_chunk(idx)
                        if out:
                            yield out
                    yield sse({**template, "choices": [choice]})
        # upstream ended without [DONE]: still flush so nothing is lost
        for idx in list(redactors):
            out = flush_chunk(idx)
            if out:
                yield out
    except httpx.HTTPError as e:
        log.error("upstream stream broke: %s", type(e).__name__)
        yield sse({"error": {"type": "gateway_error", "code": "upstream_stream_error", "message": "Upstream stream interrupted"}})
    finally:
        await upstream.aclose()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return gateway_error(400, "invalid_json", "Request body must be JSON")
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        return gateway_error(400, "invalid_request", "messages array is required")

    http: httpx.AsyncClient = request.app.state.http
    stream = bool(body.get("stream"))

    try:
        req = http.build_request("POST", UPSTREAM_URL, json=body, headers=_upstream_headers())
        upstream = await http.send(req, stream=stream)
    except httpx.TimeoutException:
        return gateway_error(504, "upstream_timeout", "Model provider timed out")
    except httpx.HTTPError:
        return gateway_error(502, "upstream_unavailable", "Model provider unavailable")

    if upstream.status_code >= 400:
        status = upstream.status_code
        await upstream.aclose()
        log.warning("upstream returned %s", status)
        return gateway_error(status if status in (400, 401, 403, 404, 429) else 502, "upstream_error", "Model provider returned an error")

    if not stream:
        data = upstream.json()
        for choice in data.get("choices", []):
            msg = choice.get("message") or {}
            if isinstance(msg.get("content"), str):
                msg["content"] = redact(msg["content"])
        return JSONResponse(data)

    return StreamingResponse(
        redact_stream(upstream),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )


@app.get("/healthz")
async def healthz():
    return {"ok": True}
