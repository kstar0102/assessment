"""Run:  pytest -q task4_llm_router"""

import asyncio
import sqlite3
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import app as gw  # noqa: E402
from limiter import RateLimited, RequestTooLarge, SlidingWindowLimiter, estimate_tokens  # noqa: E402
from mock_providers import make_app  # noqa: E402
from router import DEFAULT_TIMEOUT_S, ModelRouter, Provider  # noqa: E402


# ---- limiter ---------------------------------------------------------------

@pytest.fixture
def limiter(tmp_path):
    return SlidingWindowLimiter(str(tmp_path / "rl.db"), limit_tokens=50_000, window_seconds=60)


async def test_db_is_on_disk(limiter):
    await limiter.reserve("t", 100)
    assert Path(limiter.db_path).exists()
    rows = sqlite3.connect(limiter.db_path).execute("SELECT COUNT(*) FROM token_usage").fetchone()[0]
    assert rows == 1


async def test_sliding_window_and_eviction(limiter):
    t0 = 1_000_000.0
    await limiter.reserve("a", 30_000, now=t0)
    await limiter.reserve("a", 20_000, now=t0 + 30)
    with pytest.raises(RateLimited) as e:
        await limiter.reserve("a", 1, now=t0 + 59)
    assert e.value.retry_after == pytest.approx(1.0)       # first row expires at t0+60
    await limiter.reserve("a", 30_000, now=t0 + 60.01)      # first row slid out
    with pytest.raises(RateLimited):
        await limiter.reserve("a", 1, now=t0 + 61)
    assert await limiter.usage("a", now=t0 + 61) == 50_000
    # expired rows are physically deleted, not just ignored
    n = sqlite3.connect(limiter.db_path).execute("SELECT COUNT(*) FROM token_usage WHERE ts <= ?", (t0,)).fetchone()[0]
    assert n == 0


async def test_tenants_are_isolated(limiter):
    await limiter.reserve("a", 50_000)
    await limiter.reserve("b", 50_000)
    with pytest.raises(RateLimited):
        await limiter.reserve("a", 1)


async def test_commit_reconciles_and_release_refunds(limiter):
    r = await limiter.reserve("a", 40_000)
    await limiter.commit(r, 1_000)
    assert await limiter.usage("a") == 1_000
    r2 = await limiter.reserve("a", 49_000)
    await limiter.release(r2)
    assert await limiter.usage("a") == 1_000


async def test_too_large(limiter):
    with pytest.raises(RequestTooLarge):
        await limiter.reserve("a", 50_001)


async def test_concurrent_reservations_never_overshoot(limiter):
    async def one():
        try:
            await limiter.reserve("hot", 5_000)
            return True
        except RateLimited:
            return False

    results = await asyncio.gather(*[one() for _ in range(40)])
    assert sum(results) == 10
    assert await limiter.usage("hot") == 50_000


def test_estimate_tokens():
    assert estimate_tokens({"messages": [{"role": "user", "content": "x" * 400}], "max_tokens": 100}) == 104 + 100


# ---- router ----------------------------------------------------------------

class HostRouter(httpx.AsyncBaseTransport):
    def __init__(self, apps):
        self.t = {h: httpx.ASGITransport(app=a) for h, a in apps.items()}

    async def handle_async_request(self, request):
        return await self.t[request.url.host].handle_async_request(request)


PROVIDERS = [Provider("primary", "http://primary/v1/chat/completions", "p-model"),
             Provider("secondary", "http://secondary/v1/chat/completions", "s-model")]


def make_router(primary_mode, secondary_mode="ok", timeout=0.3):
    p, s = make_app("primary", primary_mode), make_app("secondary", secondary_mode)
    p.state.slow_s = s.state.slow_s = 2
    client = httpx.AsyncClient(transport=HostRouter({"primary": p, "secondary": s}))
    return ModelRouter(PROVIDERS, client, timeout_s=timeout), p, s, client


def test_default_timeout_is_3000ms():
    assert DEFAULT_TIMEOUT_S == 3.0


@pytest.mark.parametrize("mode", ["429", "500", "slow", "garbage"])
async def test_failover(mode):
    router, p, s, _ = make_router(mode)
    t0 = time.perf_counter()
    res = await router.complete({"messages": [{"role": "user", "content": "hi"}]}, "rid")
    assert res.provider == "secondary"
    assert res.data["model"] == "s-model"
    assert p.state.hits == 1 and s.state.hits == 1
    if mode == "slow":
        assert time.perf_counter() - t0 < 1.0  # cut at timeout, not at the 2s sleep


async def test_no_failover_on_client_error():
    router, _, s, _ = make_router("400")
    with pytest.raises(gw.GatewayError) as e:
        await router.complete({"messages": []}, "rid")
    assert e.value.status == 400 and s.state.hits == 0


async def test_primary_ok_skips_secondary():
    router, _, s, _ = make_router("ok")
    assert (await router.complete({"messages": []}, "rid")).provider == "primary"
    assert s.state.hits == 0


# ---- full app ----------------------------------------------------------------

@pytest.fixture
async def api(tmp_path):
    def build(primary_mode="ok", secondary_mode="ok", limit=50_000):
        p, s = make_app("primary", primary_mode), make_app("secondary", secondary_mode)
        p.state.slow_s = s.state.slow_s = 2
        http = httpx.AsyncClient(transport=HostRouter({"primary": p, "secondary": s}))
        gw.app.state.limiter = SlidingWindowLimiter(str(tmp_path / "gw.db"), limit_tokens=limit)
        gw.app.state.router = ModelRouter(PROVIDERS, http, timeout_s=0.3)
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app, raise_app_exceptions=False), base_url="http://gw")
    return build


AUTH = {"Authorization": "Bearer tenant-key-1"}
BODY = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 100}


def assert_sanitised(r, code):
    body = r.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"type", "code", "message", "request_id"}
    assert body["error"]["code"] == code
    for leak in ("Traceback", "infer.py", "shard", "org-123", "internal-v9", "primary", "secondary"):
        assert leak not in r.text


async def test_api_ok(api):
    async with api() as c:
        r = await c.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert r.status_code == 200
    assert r.headers["x-gateway-provider"] == "primary"


async def test_api_fallback_header(api):
    async with api(primary_mode="429") as c:
        r = await c.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert r.json()["choices"][0]["message"]["content"] == "hello from secondary"
    assert r.headers["x-gateway-fallback"] == "true"


async def test_api_all_down_is_sanitised(api):
    async with api(primary_mode="500", secondary_mode="slow") as c:
        r = await c.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert r.status_code == 503
    assert_sanitised(r, "upstream_unavailable")
    assert await gw.app.state.limiter.usage(gw.tenant_from(type("R", (), {"headers": AUTH})())) == 0  # refunded


async def test_api_upstream_400_sanitised(api):
    async with api(primary_mode="400") as c:
        r = await c.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert r.status_code == 400
    assert_sanitised(r, "invalid_request")


async def test_api_rate_limit(api):
    async with api(limit=300) as c:
        ok = await c.post("/v1/chat/completions", json={**BODY, "max_tokens": 250}, headers=AUTH)
        assert ok.status_code == 200   # usage reconciled down to 15 tokens
        r1 = await c.post("/v1/chat/completions", json={**BODY, "max_tokens": 250}, headers=AUTH)
        assert r1.status_code == 200
        r2 = await c.post("/v1/chat/completions", json={**BODY, "max_tokens": 290}, headers=AUTH)
    assert r2.status_code == 429
    assert int(r2.headers["retry-after"]) >= 1
    assert_sanitised(r2, "rate_limited")


async def test_api_auth_and_validation(api):
    async with api() as c:
        r = await c.post("/v1/chat/completions", json=BODY)
        assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"
        r = await c.post("/v1/chat/completions", json={"messages": []}, headers=AUTH)
        assert_sanitised(r, "invalid_request")
        r = await c.post("/v1/chat/completions", content=b"{bad", headers={**AUTH, "content-type": "application/json"})
        assert r.status_code == 400


async def test_api_unexpected_crash_is_sanitised(api, monkeypatch):
    async with api() as c:
        async def boom(*a, **k):
            raise RuntimeError("db password=hunter2 at /etc/secret")
        monkeypatch.setattr(gw.app.state.router, "complete", boom)
        r = await c.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert r.status_code == 500
    assert "hunter2" not in r.text
    assert r.json()["error"]["code"] == "internal_error"
