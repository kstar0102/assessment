"""
Task 4: LLM gateway with per tenant token rate limiting and model failover.

POST /v1/chat/completions
  Authorization: Bearer <tenant api key>

  1. Tenant = sha256(api key). Raw keys are never stored.
  2. Validate body, estimate tokens, reserve them in SQLite (429 if over).
  3. Route: primary, then secondary on 429 / 5xx / timeout (3000ms).
  4. Reconcile the reservation with real usage, or release it on failure.
  5. Every error, including unexpected crashes, uses one sanitised shape:
       {"error": {"type": "gateway_error", "code": "...", "message": "...", "request_id": "..."}}

Env
  RATE_DB_PATH          ./data/gateway.db
  TOKENS_PER_MINUTE     50000
  PROVIDER_TIMEOUT_MS   3000
  PRIMARY_URL / PRIMARY_MODEL / PRIMARY_API_KEY
  SECONDARY_URL / SECONDARY_MODEL / SECONDARY_API_KEY

Run:  uvicorn app:app --port 8082
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

try:
    from .limiter import RateLimited, RequestTooLarge, SlidingWindowLimiter, estimate_tokens
    from .router import GatewayError, ModelRouter, Provider
except ImportError:
    from limiter import RateLimited, RequestTooLarge, SlidingWindowLimiter, estimate_tokens
    from router import GatewayError, ModelRouter, Provider

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("llm-gateway")


# ---- request schema -------------------------------------------------------

class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: Any = None


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    messages: list[Message] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0, le=32_000)
    stream: bool = False


# ---- error helpers --------------------------------------------------------

def error_response(request: Request, status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    rid = getattr(request.state, "request_id", None) or uuid.uuid4().hex
    return JSONResponse(
        {"error": {"type": "gateway_error", "code": code, "message": message, "request_id": rid}},
        status_code=status,
        headers={"x-request-id": rid, **(headers or {})},
    )


def build_providers() -> list[Provider]:
    return [
        Provider("primary", os.environ.get("PRIMARY_URL", "http://127.0.0.1:9201/v1/chat/completions"),
                 os.environ.get("PRIMARY_MODEL", "primary-model"), os.environ.get("PRIMARY_API_KEY", "")),
        Provider("secondary", os.environ.get("SECONDARY_URL", "http://127.0.0.1:9202/v1/chat/completions"),
                 os.environ.get("SECONDARY_MODEL", "secondary-model"), os.environ.get("SECONDARY_API_KEY", "")),
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.limiter = SlidingWindowLimiter(
        os.environ.get("RATE_DB_PATH", "./data/gateway.db"),
        limit_tokens=int(os.environ.get("TOKENS_PER_MINUTE", 50_000)),
    )
    app.state.http = httpx.AsyncClient(limits=httpx.Limits(max_connections=200, max_keepalive_connections=50))
    app.state.router = ModelRouter(build_providers(), app.state.http, timeout_s=int(os.environ.get("PROVIDER_TIMEOUT_MS", 3000)) / 1000)
    yield
    await app.state.http.aclose()


app = FastAPI(title="LLM Gateway Router", lifespan=lifespan)


@app.middleware("http")
async def request_id_mw(request: Request, call_next):
    request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def on_validation(request: Request, exc: RequestValidationError):
    fields = sorted({".".join(str(p) for p in e["loc"] if p != "body") for e in exc.errors()})
    return error_response(request, 400, "invalid_request", f"Invalid request fields: {', '.join(fields) or 'body'}")


@app.exception_handler(StarletteHTTPException)
async def on_http(request: Request, exc: StarletteHTTPException):
    return error_response(request, exc.status_code, "http_error", "Not found" if exc.status_code == 404 else "Request error")


@app.exception_handler(Exception)
async def on_unhandled(request: Request, exc: Exception):
    log.exception("rid=%s unhandled error", getattr(request.state, "request_id", "?"))
    return error_response(request, 500, "internal_error", "An internal error occurred.")


def tenant_from(request: Request) -> str | None:
    scheme, _, key = (request.headers.get("authorization") or "").partition(" ")
    if scheme.lower() != "bearer" or not key.strip():
        return None
    return hashlib.sha256(key.strip().encode()).hexdigest()[:32]


@app.post("/v1/chat/completions")
async def chat_completions(body: CompletionRequest, request: Request):
    rid = request.state.request_id
    tenant = tenant_from(request)
    if tenant is None:
        return error_response(request, 401, "unauthorized", "Missing or invalid API key.")
    if body.stream:
        return error_response(request, 400, "stream_not_supported", "Streaming is not supported on this endpoint.")

    limiter: SlidingWindowLimiter = request.app.state.limiter
    router: ModelRouter = request.app.state.router
    payload = body.model_dump(exclude_none=True)
    estimate = estimate_tokens(payload)

    try:
        reservation = await limiter.reserve(tenant, estimate)
    except RequestTooLarge:
        return error_response(request, 413, "request_too_large", f"Request needs ~{estimate} tokens, above the per minute limit.")
    except RateLimited as e:
        retry = max(1, int(e.retry_after + 0.999))
        return error_response(
            request, 429, "rate_limited", "Token rate limit exceeded for this API key.",
            headers={"retry-after": str(retry), "x-ratelimit-limit-tokens": str(e.limit),
                     "x-ratelimit-remaining-tokens": str(max(0, e.limit - e.used))},
        )

    try:
        result = await router.complete(payload, rid)
    except GatewayError as e:
        await limiter.release(reservation)
        headers = {"retry-after": str(int(e.retry_after))} if e.retry_after else None
        return error_response(request, e.status, e.code, e.message, headers)
    except asyncio.CancelledError:
        # client went away mid flight; the provider may still bill us, so keep the estimate
        raise
    except Exception:
        await limiter.release(reservation)
        raise

    usage = result.data.get("usage") or {}
    actual = usage.get("total_tokens")
    if isinstance(actual, int) and actual >= 0:
        await limiter.commit(reservation, actual)
        remaining = reservation.remaining + reservation.tokens - actual
    else:
        remaining = reservation.remaining

    return JSONResponse(
        result.data,
        headers={
            "x-request-id": rid,
            "x-gateway-provider": result.provider,
            "x-gateway-fallback": "true" if len(result.attempts) > 1 else "false",
            "x-ratelimit-limit-tokens": str(limiter.limit),
            "x-ratelimit-remaining-tokens": str(max(0, remaining)),
        },
    )


@app.get("/healthz")
async def healthz():
    return {"ok": True}
