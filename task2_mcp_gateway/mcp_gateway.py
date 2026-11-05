"""
Task 2: MCP security gateway (reverse proxy).

  agent  --HTTP JSON-RPC-->  gateway  --HTTP JSON-RPC-->  downstream MCP server

Flow per request
  1. Authenticate: Authorization: Bearer <JWT>, role claim must be admin|viewer.
     Failure is HTTP 401 with a JSON-RPC error body.
  2. Parse the JSON-RPC envelope (single message or batch). Malformed JSON is
     -32700, a bad envelope is -32600.
  3. Authorise per message:
       tools/list          forward as is
       tools/call admin_*  role must be admin, else -32001 and the downstream
                           server is never called
       anything else       forward (initialize, ping, notifications...)
  4. Forward allowed messages. The user's token is NOT passed downstream, the
     gateway sends a trusted identity header instead.
  5. Relay the response: status, content type, Mcp-Session-Id. SSE responses
     from streamable HTTP servers are streamed straight through.

Run:  uvicorn mcp_gateway:app --port 8080
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

try:
    from .auth import AuthError, parse_bearer, verify_token
except ImportError:  # run as a script from this folder
    from auth import AuthError, parse_bearer, verify_token

log = logging.getLogger("mcp-gateway")

DOWNSTREAM_URL = os.environ.get("DOWNSTREAM_MCP_URL", "http://127.0.0.1:9000/mcp")
JWT_SECRET = os.environ.get("GATEWAY_JWT_SECRET", "dev-secret-change-me")
MAX_BODY_BYTES = int(os.environ.get("GATEWAY_MAX_BODY", 1_000_000))

# JSON-RPC codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INTERNAL_ERROR = -32603
UNAUTHORIZED = -32001
BAD_GATEWAY = -32002

FORWARD_REQ_HEADERS = {"content-type", "accept", "mcp-session-id", "mcp-protocol-version", "last-event-id"}
FORWARD_RESP_HEADERS = {"mcp-session-id", "mcp-protocol-version"}


def rpc_error(msg_id: Any, code: int, message: str, data: Any = None) -> dict:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": err}


def is_valid_envelope(msg: Any) -> bool:
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return False
    if "method" in msg:  # request or notification
        if not isinstance(msg["method"], str):
            return False
        if "params" in msg and not isinstance(msg["params"], (dict, list)):
            return False
        return True
    # a client may also POST responses (e.g. answering a server elicitation)
    return "id" in msg and ("result" in msg or "error" in msg)


def authorise(msg: dict, role: str) -> dict | None:
    """Return a JSON-RPC error if this message must be blocked, else None."""
    if msg.get("method") != "tools/call":
        return None
    params = msg.get("params")
    name = params.get("name") if isinstance(params, dict) else None
    if not isinstance(name, str) or not name:
        return rpc_error(msg.get("id"), -32602, "Invalid params: tools/call requires params.name")
    # casefold so ADMIN_reset_key can't sneak past a case sensitive check
    if name.casefold().startswith("admin_") and role != "admin":
        return rpc_error(msg.get("id"), UNAUTHORIZED, "Unauthorized Tool Call", {"tool": name, "required_role": "admin"})
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
    yield
    await app.state.http.aclose()


app = FastAPI(title="MCP Security Gateway", lifespan=lifespan)


def _downstream_headers(request: Request, claims: dict, request_id: str) -> dict:
    headers = {k: v for k, v in request.headers.items() if k.lower() in FORWARD_REQ_HEADERS}
    headers.setdefault("content-type", "application/json")
    headers.setdefault("accept", "application/json, text/event-stream")
    headers["x-gateway-user"] = str(claims.get("sub", ""))
    headers["x-gateway-role"] = claims["role"]
    headers["x-request-id"] = request_id
    return headers


@app.post("/mcp")
async def mcp_proxy(request: Request) -> Response:
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex

    # 1. authenticate
    try:
        claims = verify_token(parse_bearer(request.headers.get("authorization")), JWT_SECRET)
    except AuthError as e:
        log.info("auth rejected rid=%s reason=%s", request_id, e)
        return JSONResponse(
            rpc_error(None, UNAUTHORIZED, "Unauthorized"),
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        )
    role = claims["role"]

    # 2. parse
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        return JSONResponse(rpc_error(None, INVALID_REQUEST, "Request too large"), status_code=413)
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(rpc_error(None, PARSE_ERROR, "Parse error"), status_code=400)

    is_batch = isinstance(payload, list)
    messages = payload if is_batch else [payload]
    if is_batch and not messages:
        return JSONResponse(rpc_error(None, INVALID_REQUEST, "Invalid Request: empty batch"), status_code=400)

    # 3. validate + authorise each message
    blocked: list[dict] = []
    allowed: list[dict] = []
    for msg in messages:
        if not is_valid_envelope(msg):
            mid = msg.get("id") if isinstance(msg, dict) else None
            blocked.append(rpc_error(mid, INVALID_REQUEST, "Invalid Request"))
            continue
        denial = authorise(msg, role)
        if denial is None:
            allowed.append(msg)
        else:
            log.warning("blocked rid=%s sub=%s role=%s tool=%s", request_id, claims.get("sub"), role, msg["params"].get("name") if isinstance(msg.get("params"), dict) else None)
            if "id" in msg:  # notifications never get a response
                blocked.append(denial)

    if not allowed:
        if not blocked:
            return Response(status_code=202)
        body = blocked if is_batch else blocked[0]
        return JSONResponse(body, status_code=200)

    headers = _downstream_headers(request, claims, request_id)
    http: httpx.AsyncClient = request.app.state.http

    # 4a. single message: true streaming pass through (supports SSE)
    if not is_batch:
        try:
            upstream_req = http.build_request("POST", DOWNSTREAM_URL, json=allowed[0], headers=headers)
            upstream = await http.send(upstream_req, stream=True)
        except httpx.HTTPError as e:
            log.error("downstream unreachable rid=%s err=%s", request_id, type(e).__name__)
            return JSONResponse(rpc_error(allowed[0].get("id"), BAD_GATEWAY, "Downstream MCP server unavailable"), status_code=502)

        resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() in FORWARD_RESP_HEADERS}
        resp_headers["x-request-id"] = request_id
        content_type = upstream.headers.get("content-type", "application/json")

        async def body_iter():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(body_iter(), status_code=upstream.status_code, headers=resp_headers, media_type=content_type)

    # 4b. batch: forward allowed subset, merge with locally generated errors
    headers["accept"] = "application/json"
    try:
        upstream = await http.post(DOWNSTREAM_URL, json=allowed, headers=headers)
    except httpx.HTTPError as e:
        log.error("downstream unreachable rid=%s err=%s", request_id, type(e).__name__)
        errs = [rpc_error(m.get("id"), BAD_GATEWAY, "Downstream MCP server unavailable") for m in allowed if "id" in m]
        return JSONResponse(blocked + errs, status_code=502)

    downstream_results: list = []
    if upstream.content:
        try:
            parsed = upstream.json()
            downstream_results = parsed if isinstance(parsed, list) else [parsed]
        except ValueError:
            errs = [rpc_error(m.get("id"), INTERNAL_ERROR, "Invalid downstream response") for m in allowed if "id" in m]
            return JSONResponse(blocked + errs, status_code=502)

    merged = downstream_results + blocked
    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() in FORWARD_RESP_HEADERS}
    resp_headers["x-request-id"] = request_id
    if not merged:
        return Response(status_code=202, headers=resp_headers)
    return JSONResponse(merged, status_code=200, headers=resp_headers)


@app.get("/healthz")
async def healthz():
    return {"ok": True}
