"""Run:  pytest -q task2_mcp_gateway"""

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import mcp_gateway as gateway  # noqa: E402
import mock_mcp_server  # noqa: E402
from auth import make_token  # noqa: E402

SECRET = "test-secret"
ADMIN = make_token("alice", "admin", SECRET)
VIEWER = make_token("bob", "viewer", SECRET)


@pytest.fixture
async def client(monkeypatch):
    monkeypatch.setattr(gateway, "JWT_SECRET", SECRET)
    monkeypatch.setattr(gateway, "DOWNSTREAM_URL", "http://downstream/mcp")
    mock_mcp_server.app.state.calls = []
    gateway.app.state.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_mcp_server.app))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url="http://gw") as c:
        yield c
    await gateway.app.state.http.aclose()


def rpc(method, params=None, mid=1):
    return {"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}}


def auth(token):
    return {"Authorization": f"Bearer {token}"}


async def test_missing_token(client):
    r = await client.post("/mcp", json=rpc("tools/list"))
    assert r.status_code == 401
    assert r.json()["error"]["code"] == -32001


@pytest.mark.parametrize("header", ["Bearer", "Basic abc", "Bearer not.a.jwt", f"Bearer {make_token('x', 'admin', 'wrong-secret')}", f"Bearer {make_token('x', 'admin', SECRET, ttl=-120)}"])
async def test_bad_tokens(client, header):
    r = await client.post("/mcp", json=rpc("tools/list"), headers={"Authorization": header})
    assert r.status_code == 401


async def test_tools_list_forwarded_for_viewer(client):
    r = await client.post("/mcp", json=rpc("tools/list"), headers=auth(VIEWER))
    assert r.status_code == 200
    names = [t["name"] for t in r.json()["result"]["tools"]]
    assert "admin_reset_key" in names
    assert r.headers["mcp-session-id"] == "mock-session-1"


async def test_viewer_can_call_normal_tool(client):
    r = await client.post("/mcp", json=rpc("tools/call", {"name": "get_customer_record", "arguments": {}}), headers=auth(VIEWER))
    assert r.json()["result"]["content"][0]["text"] == "ran get_customer_record"
    call = mock_mcp_server.app.state.calls[0]
    assert call["user"] == "bob"
    assert call["auth"] is None  # user token never leaks downstream


@pytest.mark.parametrize("tool", ["admin_reset_key", "ADMIN_reset_key", "Admin_wipe"])
async def test_viewer_blocked_from_admin_tool(client, tool):
    r = await client.post("/mcp", json=rpc("tools/call", {"name": tool}, mid=7), headers=auth(VIEWER))
    body = r.json()
    assert body["id"] == 7
    assert body["error"]["code"] == -32001
    assert body["error"]["message"] == "Unauthorized Tool Call"
    assert mock_mcp_server.app.state.calls == []  # downstream never hit


async def test_admin_can_call_admin_tool(client):
    r = await client.post("/mcp", json=rpc("tools/call", {"name": "admin_reset_key"}), headers=auth(ADMIN))
    assert r.json()["result"]["isError"] is False
    assert mock_mcp_server.app.state.calls[0]["tool"] == "admin_reset_key"


async def test_parse_error(client):
    r = await client.post("/mcp", content=b"{not json", headers={**auth(VIEWER), "content-type": "application/json"})
    assert r.json()["error"]["code"] == -32700


@pytest.mark.parametrize("body", [{"id": 1, "method": "tools/list"}, {"jsonrpc": "2.0", "id": 1, "method": 5}, [], "hello"])
async def test_invalid_request(client, body):
    r = await client.post("/mcp", json=body, headers=auth(VIEWER))
    assert r.json()["error"]["code"] == -32600


async def test_tools_call_without_name(client):
    r = await client.post("/mcp", json=rpc("tools/call", {}), headers=auth(VIEWER))
    assert r.json()["error"]["code"] == -32602


async def test_batch_mixed(client):
    batch = [
        rpc("tools/list", mid=1),
        rpc("tools/call", {"name": "admin_reset_key"}, mid=2),
        rpc("tools/call", {"name": "get_customer_record"}, mid=3),
    ]
    r = await client.post("/mcp", json=batch, headers=auth(VIEWER))
    by_id = {m["id"]: m for m in r.json()}
    assert "result" in by_id[1]
    assert by_id[2]["error"]["code"] == -32001
    assert "result" in by_id[3]
    assert [c["tool"] for c in mock_mcp_server.app.state.calls] == ["get_customer_record"]


async def test_notification_forwarded(client):
    r = await client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=auth(VIEWER))
    assert r.status_code == 202


async def test_downstream_down(client, monkeypatch):
    monkeypatch.setattr(gateway, "DOWNSTREAM_URL", "http://127.0.0.1:1/mcp")
    gateway.app.state.http = httpx.AsyncClient()
    r = await client.post("/mcp", json=rpc("tools/list"), headers=auth(VIEWER))
    assert r.status_code == 502
    assert "Traceback" not in r.text
