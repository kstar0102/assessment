# AI Field Developer / Solutions Engineer Assessment

All four tasks in Python 3.11+. Each one is runnable on its own, ships with a mock to run against, and has a test suite. 284 tests, all green.

```
task1_mcp_server/      stdio MCP server, strict Pydantic validation
task2_mcp_gateway/     JWT auth + tool level authorisation reverse proxy
task3_llm_guardrail/   streaming PII redaction gateway
task4_llm_router/      SQLite sliding window rate limiter + model failover
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q            # runs everything
```

## Task 1: MCP server

```bash
cd task1_mcp_server
python server.py                 # speaks JSON-RPC on stdio
pytest -q                        # black box tests over real pipes
```

Add it to Claude Desktop or any MCP client:

```json
{ "mcpServers": { "customers": { "command": "python", "args": ["/abs/path/task1_mcp_server/server.py"] } } }
```

Key decisions

* **Stdout isolation at the fd level.** The real stdout fd is duplicated for the transport, then fd 1 and `sys.stdout` are pointed at stderr. A stray `print()` or a noisy C library can't corrupt the stream. The test proves it with a canary print.
* **Strict models.** `extra="forbid"`, `strict=True` so `"10"` isn't a number and `true` isn't an amount. Amount must be positive, finite, at most 2dp. Reason is trimmed before the length check, so ten spaces don't pass.
* **Error mapping.** Bad input or unknown tool is JSON-RPC `-32602` with field level detail (raw input never echoed). Business failures like unknown customer come back as `isError: true` results, which is what the MCP spec wants so the model can react. Crashes are `-32603` with a generic message.
* I register the `tools/call` handler directly instead of using `@server.call_tool()`, because the decorator turns every exception into an `isError` result and you'd never see a real protocol error.

## Task 2: MCP security gateway

```bash
cd task2_mcp_gateway
uvicorn mock_mcp_server:app --port 9000 &
DOWNSTREAM_MCP_URL=http://127.0.0.1:9000/mcp uvicorn mcp_gateway:app --port 8080 &

VIEWER=$(python auth.py viewer)
curl -s localhost:8080/mcp -H "Authorization: Bearer $VIEWER" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_reset_key"}}'
# {"jsonrpc":"2.0","id":1,"error":{"code":-32001,"message":"Unauthorized Tool Call",...}}
```

* Tokens are HS256 JWTs verified with the stdlib. `alg` is pinned so `alg: none` tricks fail, signatures use constant time compare, expiry is checked, role must be `admin` or `viewer`.
* `tools/list` and everything else is forwarded. `tools/call` on `admin_*` needs admin, otherwise `-32001` and the downstream is never touched (tests assert zero downstream hits). The prefix check is case insensitive so `ADMIN_reset_key` can't sneak past.
* Handles parse errors (`-32700`), bad envelopes (`-32600`), notifications, and batches where some calls are allowed and some blocked.
* The user's token is stripped before forwarding. Downstream gets `x-gateway-user` and `x-gateway-role` instead. `Mcp-Session-Id` is passed both ways and SSE responses stream straight through.

## Task 3: Streaming PII guardrail

```bash
cd task3_llm_guardrail
uvicorn mock_llm:app --port 9100 &
uvicorn guardrail_gateway:app --port 8081 &
curl -N localhost:8081/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Point it at a real provider with `UPSTREAM_URL` and `UPSTREAM_API_KEY` (any OpenAI compatible endpoint).

* **The hard bit is PII split across chunks**, e.g. `"jo"` then `"hn@acme.io"`. The redactor keeps a tiny holdback tail: only the trailing word or digit run that could still become a match. Everything before it is final, so it's redacted and sent straight away.
* Memory is bounded by the longest PII token, not the response. A hard cap (256 chars) forces a flush if someone streams a giant unbroken token.
* Content free chunks (role, finish reason, tool calls) pass through untouched, so TTFT isn't held up. The tail is flushed before `finish_reason` and `[DONE]`.
* Cards are Luhn checked so order numbers survive. SSNs skip invalid SSA ranges.
* Tests fuzz 200 random chunkings plus char by char and assert the streamed output is byte identical to redacting the full text in one go. A timing test proves the first chunk leaves well before the upstream finishes.

## Task 4: Rate limiter + fallback router

```bash
cd task4_llm_router
MOCK_MODE=slow MOCK_NAME=primary   uvicorn mock_providers:app --port 9201 &
MOCK_MODE=ok   MOCK_NAME=secondary uvicorn mock_providers:app --port 9202 &
uvicorn app:app --port 8082 &
curl -si localhost:8082/v1/chat/completions -H "Authorization: Bearer my-tenant-key" \
  -d '{"messages":[{"role":"user","content":"hi"}]}'
# after ~3s: x-gateway-provider: secondary, x-gateway-fallback: true
```

Try `MOCK_MODE=429`, `500`, `400` or `garbage` on the primary.

* **Sliding log in SQLite on disk** (`./data/gateway.db`, WAL mode). One row per request. Usage is the sum over the last 60s, so no burst at window edges.
* **Race safe.** Check and insert happen inside `BEGIN IMMEDIATE`, so concurrent requests can't all see the same headroom. The test fires 40 concurrent 5k token requests at a 50k limit and exactly 10 get through. Works across worker processes too, since the lock lives in the DB file.
* **Reserve, then reconcile.** Estimate (prompt chars / 4 + `max_tokens`) is reserved up front, swapped for real `usage.total_tokens` after, and refunded if no provider served it.
* **Eviction.** Expired rows are deleted on every reserve via an index on `ts`, so the table stays around one window in size. `Retry-After` is calculated from exactly when enough old rows will slide out.
* **Timeout.** The whole attempt runs under `asyncio.timeout(3.0)`, not httpx's per phase timeouts, so a slow trickle can't stretch it. On expiry the request is cancelled and the connection dropped, so a late response can't race in.
* **Failover** on 429, 5xx, timeout, network error or malformed body. A normal 4xx doesn't fail over since the backup would reject it too.
* **Sanitised errors.** Every error, including unhandled crashes and validation failures, is `{"error": {"type", "code", "message", "request_id"}}`. Tests assert upstream tracebacks, org ids, model names and provider names never leak.

Config: `RATE_DB_PATH`, `TOKENS_PER_MINUTE`, `PROVIDER_TIMEOUT_MS`, `PRIMARY_URL`, `PRIMARY_MODEL`, `PRIMARY_API_KEY`, and the same for `SECONDARY_`.

## What I'd add next in production

* Task 2: JWKS / OIDC instead of a shared secret, and optionally hide `admin_*` tools from `tools/list` for viewers.
* Task 3: pluggable detectors (names, phone numbers, IBANs via a NER model off the hot path) and per tenant policies.
* Task 4: Redis for very high throughput across many gateway nodes, a circuit breaker so a dead primary isn't retried on every request, and streaming support through the router.
