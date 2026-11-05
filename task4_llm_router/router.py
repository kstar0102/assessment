"""
Model router with timeout bound failover.

For each provider in priority order:
  * The whole attempt (connect, send, read body) runs under asyncio.timeout,
    so a slow trickle can't dodge the 3000ms budget the way per phase httpx
    timeouts can. On expiry the request task is cancelled and the connection
    is dropped, so no late response can race in and get used.
  * 429, 5xx, timeouts and network errors mean "try the next provider".
  * Other 4xx means the request itself is bad. Failing over would just get
    the same answer, so we stop and return a sanitised client error.

Nothing from upstream (bodies, headers, exception text) ever reaches the
caller. It's logged server side with the request id instead.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx

log = logging.getLogger("llm-router")

DEFAULT_TIMEOUT_S = 3.0


@dataclass(frozen=True)
class Provider:
    name: str
    url: str
    model: str
    api_key: str = ""


class GatewayError(Exception):
    def __init__(self, status: int, code: str, message: str, retry_after: float | None = None):
        super().__init__(code)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after = retry_after


@dataclass
class Attempt:
    provider: str
    outcome: str
    ms: int


@dataclass
class RouteResult:
    provider: str
    data: dict
    attempts: list[Attempt] = field(default_factory=list)


class ModelRouter:
    def __init__(self, providers: list[Provider], client: httpx.AsyncClient, timeout_s: float = DEFAULT_TIMEOUT_S):
        if not providers:
            raise ValueError("at least one provider required")
        self.providers = providers
        self.client = client
        self.timeout_s = timeout_s

    async def _call(self, p: Provider, body: dict, request_id: str) -> httpx.Response:
        headers = {"content-type": "application/json", "x-request-id": request_id}
        if p.api_key:
            headers["authorization"] = f"Bearer {p.api_key}"
        payload = {**body, "model": p.model, "stream": False}
        async with asyncio.timeout(self.timeout_s):
            resp = await self.client.post(p.url, json=payload, headers=headers)
            await resp.aread()
            return resp

    async def complete(self, body: dict, request_id: str) -> RouteResult:
        attempts: list[Attempt] = []
        for p in self.providers:
            t0 = time.perf_counter()
            ms = lambda: int((time.perf_counter() - t0) * 1000)  # noqa: E731
            try:
                resp = await self._call(p, body, request_id)
            except TimeoutError:
                attempts.append(Attempt(p.name, "timeout", ms()))
                log.warning("rid=%s provider=%s timeout after %dms, failing over", request_id, p.name, ms())
                continue
            except httpx.HTTPError as e:
                attempts.append(Attempt(p.name, f"network:{type(e).__name__}", ms()))
                log.warning("rid=%s provider=%s network error %s, failing over", request_id, p.name, type(e).__name__)
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                attempts.append(Attempt(p.name, f"http_{resp.status_code}", ms()))
                log.warning("rid=%s provider=%s status=%d, failing over", request_id, p.name, resp.status_code)
                continue

            if resp.status_code >= 400:
                attempts.append(Attempt(p.name, f"http_{resp.status_code}", ms()))
                log.info("rid=%s provider=%s rejected request status=%d body=%.200s", request_id, p.name, resp.status_code, resp.text)
                raise GatewayError(400, "invalid_request", "The model provider rejected the request.")

            try:
                data = resp.json()
                if not isinstance(data, dict) or "choices" not in data:
                    raise ValueError("bad shape")
            except ValueError:
                attempts.append(Attempt(p.name, "bad_response", ms()))
                log.error("rid=%s provider=%s returned malformed body", request_id, p.name)
                continue

            attempts.append(Attempt(p.name, "ok", ms()))
            return RouteResult(p.name, data, attempts)

        log.error("rid=%s all providers failed: %s", request_id, attempts)
        raise GatewayError(503, "upstream_unavailable", "All model providers are currently unavailable. Please retry shortly.", retry_after=5)
