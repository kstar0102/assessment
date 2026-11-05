"""
Mock downstream MCP server (streamable HTTP style, JSON responses).
Counts tools/call hits so tests can prove blocked calls never arrive.

Run:  uvicorn mock_mcp_server:app --port 9000
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

app = FastAPI(title="Mock MCP server")
app.state.calls = []

TOOLS = [
    {"name": "get_customer_record", "description": "Read a customer", "inputSchema": {"type": "object", "properties": {"customer_id": {"type": "string"}}}},
    {"name": "admin_reset_key", "description": "Rotate an API key", "inputSchema": {"type": "object", "properties": {"key_id": {"type": "string"}}}},
]


def handle(msg: dict, request: Request) -> dict | None:
    method, mid = msg.get("method"), msg.get("id")
    if "id" not in msg:
        return None  # notification
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "mock", "version": "0"}}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        name = msg["params"]["name"]
        app.state.calls.append({"tool": name, "user": request.headers.get("x-gateway-user"), "auth": request.headers.get("authorization")})
        return {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": f"ran {name}"}], "isError": False}}
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "Method not found"}}


@app.post("/mcp")
async def mcp(request: Request):
    body = await request.json()
    if isinstance(body, list):
        out = [r for r in (handle(m, request) for m in body) if r]
        return JSONResponse(out) if out else Response(status_code=202)
    r = handle(body, request)
    return JSONResponse(r, headers={"mcp-session-id": "mock-session-1"}) if r else Response(status_code=202)
